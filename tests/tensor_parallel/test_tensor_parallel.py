# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import tempfile
import warnings
from unittest.mock import MagicMock, patch

import torch

from transformers import AutoModelForCausalLM
from transformers.distributed import tensor_parallel
from transformers.distributed.sharding_utils import DtensorShardOperation
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    ColwiseParallel,
    PackedColwiseParallel,
    PackedRowwiseParallel,
    RowwiseParallel,
)
from transformers.testing_utils import TestCasePlus, is_tensor_parallel_test


@is_tensor_parallel_test
class TestTensorParallelProperties(TestCasePlus):
    def test_tp_plan_property_setter_getter(self):
        """Test that tp_plan property can be set and retrieved correctly."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test setting empty plan
        model.tp_plan = {}
        self.assertEqual(model.tp_plan, {})

        # Test setting a valid plan
        valid_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        model.tp_plan = valid_plan
        self.assertEqual(model.tp_plan, valid_plan)

        # Test updating the plan
        model.tp_plan.update({"model.layers.*.self_attn.k_proj": "colwise"})
        expected_plan = {"model.layers.*.self_attn.q_proj": "colwise", "model.layers.*.self_attn.k_proj": "colwise"}
        self.assertEqual(model.tp_plan, expected_plan)

        # Test overriding existing entry
        model.tp_plan.update({"model.layers.*.self_attn.q_proj": "rowwise"})
        expected_plan = {
            "model.layers.*.self_attn.q_proj": "rowwise",
            "model.layers.*.self_attn.k_proj": "colwise",
        }
        self.assertEqual(model.tp_plan, expected_plan)

    def test_tp_plan_validation_invalid_style(self):
        """Test that invalid parallel styles are rejected."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        invalid_plan = {
            "layers.*.self_attn.q_proj": "invalid_style",
            "layers.*.self_attn.k_proj": "another_invalid_style",
        }
        with self.assertRaises(ValueError) as context:
            model.tp_plan = invalid_plan

        error_message = str(context.exception)
        for style in invalid_plan.values():
            self.assertIn(repr(style), error_message)
        self.assertIn("Supported styles are", error_message)

    def test_apply_tensor_parallelism_reports_all_invalid_styles(self):
        model = torch.nn.Module()
        model.tp_plan = {
            "first_layer": "invalid_style",
            "second_layer": "another_invalid_style",
        }

        with self.assertRaises(ValueError) as context:
            tensor_parallel.apply_tensor_parallelism(model, tp_mesh=None)

        error_message = str(context.exception)
        self.assertIn("'invalid_style'", error_message)
        self.assertIn("'another_invalid_style'", error_message)

    def test_tp_plan_validation_nonexistent_layer_warning(self):
        """Test that warnings are issued for non-existent layer patterns."""

        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test warning for non-existent layer pattern
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.tp_plan = {"nonexistent.*.layer": "colwise"}

            # Check that a warning was issued
            self.assertTrue(len(w) > 0)
            warning_message = str(w[0].message)
            self.assertIn("Layer pattern 'nonexistent.*.layer' does not match any parameters", warning_message)

    def test_tp_plan_valid_layer_patterns(self):
        """Test that valid layer patterns are accepted without warnings."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test valid layer patterns that should match the model structure
        valid_plans = [
            {"model.layers.*.self_attn.q_proj": "colwise"},
            {"model.layers.*.self_attn.k_proj": "rowwise"},
            {"model.layers.*.mlp.gate_proj": "colwise"},
        ]

        for plan in valid_plans:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model.tp_plan = plan

                # Filter out any warnings that are not about layer patterns
                layer_warnings = [
                    warning
                    for warning in w
                    if "Layer pattern" in str(warning.message)
                    and "does not match any parameters" in str(warning.message)
                ]

                # Should not have layer pattern warnings for valid patterns
                self.assertEqual(
                    len(layer_warnings),
                    0,
                    f"Unexpected warning for valid pattern {plan}: {[str(w.message) for w in layer_warnings]}",
                )

        # Verify the final plan was set correctly
        self.assertEqual(model.tp_plan, valid_plans[-1])

    def test_tp_plan_none_handling(self):
        """Test that None values are handled correctly."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        # Test setting None
        model.tp_plan = None
        self.assertEqual(model.tp_plan, {})

        # Test setting a plan after None
        model.tp_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        self.assertEqual(model.tp_plan, {"model.layers.*.self_attn.q_proj": "colwise"})

    def test_post_init_keeps_class_level_plans(self):
        """Class-level plans (e.g. `lm_head` on ForCausalLM classes) must survive post_init alongside the base model plan."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

        self.assertIn("lm_head", model._tp_plan)
        self.assertIn("model.layers.*.self_attn.q_proj", model._tp_plan)
        self.assertIn("lm_head", model._pp_plan)
        # The merge must not have mutated the class attribute shared by all instances
        self.assertEqual(set(type(model)._tp_plan), {"lm_head"})


@is_tensor_parallel_test
class TestTensorParallelLayer(TestCasePlus):
    class MockDeviceMesh:
        def __init__(self, world_size, rank):
            self.world_size = world_size
            self.rank = rank
            self.shape = (world_size,)
            self.ndim = 1

        def size(self):
            return self.world_size

        def get_local_rank(self):
            return self.rank

    def _get_parameter_placements(self, module, style, mesh=None):
        placements = {}
        mesh = object() if mesh is None else mesh
        with patch.object(
            tensor_parallel, "distribute_tensor", side_effect=lambda tensor, *args, **kwargs: tensor
        ) as distribute:
            for parameter_name in list(module._parameters):
                style.shard_param(module, parameter_name, mesh)
                placements[parameter_name] = distribute.call_args.args[2][0]

        return placements

    def _get_local_shape(self, global_shape, placement, world_size, rank):
        if placement.is_replicate():
            return tuple(global_shape)

        shard_dim = placement.dim
        local_size, _ = placement._local_shard_size_and_offset(global_shape[shard_dim], world_size, rank)
        local_shape = list(global_shape)
        local_shape[shard_dim] = local_size
        return tuple(local_shape)

    def _make_dtensor_shard_op(self, mesh, placement, param_shape, local_shape):
        op = object.__new__(DtensorShardOperation)
        op.device_mesh = mesh
        op.placements = (placement,)
        op.param_ndim = len(param_shape)
        op._axis0_offset = 0
        op._axis0_local_size = local_shape[0]
        return op

    def test_colwise_gather_output_rejects_indivisible_out_features(self):
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(8, 99)
        model.tp_plan = {"lm_head": "colwise_gather_output"}
        device_mesh = self.MockDeviceMesh(world_size=2, rank=0)

        with self.assertRaises(ValueError) as context:
            tensor_parallel.apply_tensor_parallelism(model, device_mesh)

        self.assertIn("lm_head", str(context.exception))
        self.assertIn("divisible", str(context.exception))

    def test_colwise_uneven_local_shapes(self):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(10, 32)))
        module.register_parameter("bias", torch.nn.Parameter(torch.empty(10)))
        placements = self._get_parameter_placements(module, ColwiseParallel())
        expected_local_sizes = (4, 4, 2)

        for rank, expected_size in enumerate(expected_local_sizes):
            weight_shape = self._get_local_shape((10, 32), placements["weight"], world_size=3, rank=rank)
            bias_shape = self._get_local_shape((10,), placements["bias"], world_size=3, rank=rank)

            self.assertEqual(weight_shape, (expected_size, 32))
            self.assertEqual(bias_shape, (expected_size,))

    def test_rowwise_uneven_local_shapes(self):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(32, 10)))
        module.register_parameter("bias", torch.nn.Parameter(torch.empty(10)))
        placements = self._get_parameter_placements(module, RowwiseParallel())
        expected_local_sizes = (4, 4, 2)

        for rank, expected_size in enumerate(expected_local_sizes):
            weight_shape = self._get_local_shape((32, 10), placements["weight"], world_size=3, rank=rank)
            bias_shape = self._get_local_shape((10,), placements["bias"], world_size=3, rank=rank)

            self.assertEqual(weight_shape, (32, expected_size))
            self.assertEqual(bias_shape, (10,))

    def test_embedding_uneven_local_shapes(self):
        rowwise_embedding = torch.nn.Embedding(10, 10)
        rowwise_placement = self._get_parameter_placements(rowwise_embedding, RowwiseParallel())["weight"]

        colwise_embedding = torch.nn.Embedding(10, 10)
        colwise_placement = self._get_parameter_placements(colwise_embedding, ColwiseParallel())["weight"]

        expected_local_sizes = (4, 4, 2)
        for rank, expected_size in enumerate(expected_local_sizes):
            rowwise_shape = self._get_local_shape((10, 10), rowwise_placement, world_size=3, rank=rank)
            colwise_shape = self._get_local_shape((10, 10), colwise_placement, world_size=3, rank=rank)

            self.assertEqual(rowwise_shape, (expected_size, 10))
            self.assertEqual(colwise_shape, (10, expected_size))

    def test_shard_tensor_shape_consistency(self):
        world_size = 4
        cases = {
            "colwise": {
                "module": torch.nn.Linear(32, 16),
                "style": ColwiseParallel(),
                "expected_shapes": {"weight": (4, 32), "bias": (4,)},
            },
            "colwise_gather_output": {
                "module": torch.nn.Linear(32, 16),
                "style": ALL_PARALLEL_STYLES["colwise_gather_output"],
                "expected_shapes": {"weight": (4, 32), "bias": (4,)},
            },
            "rowwise": {
                "module": torch.nn.Linear(32, 16),
                "style": RowwiseParallel(),
                "expected_shapes": {"weight": (16, 8), "bias": (16,)},
            },
            "embedding_rowwise": {
                "module": torch.nn.Embedding(32, 16),
                "style": ALL_PARALLEL_STYLES["embedding_rowwise"],
                "expected_shapes": {"weight": (8, 16)},
            },
            "embedding_colwise": {
                "module": torch.nn.Embedding(32, 16),
                "style": ColwiseParallel(),
                "expected_shapes": {"weight": (32, 4)},
            },
        }

        for case_name, case in cases.items():
            module = case["module"]
            placements = self._get_parameter_placements(module, case["style"])

            for parameter_name, expected_shape in case["expected_shapes"].items():
                global_shape = module._parameters[parameter_name].shape
                placement = placements[parameter_name]

                for rank in range(world_size):
                    with self.subTest(case=case_name, parameter=parameter_name, rank=rank):
                        local_shape = self._get_local_shape(global_shape, placement, world_size, rank)
                        self.assertEqual(local_shape, expected_shape)

    def test_packed_colwise_packed_and_unpacked_shapes(self):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(2, 16, 64)))
        placement = self._get_parameter_placements(module, PackedColwiseParallel())["weight"]
        packed = torch.randn(2, 16, 64)
        unpacked_expert = torch.randn(16, 64)

        self.assertEqual(placement.dim, 1)
        self.assertEqual(placement.split_factor, 2)
        for rank in range(2):
            mesh = self.MockDeviceMesh(world_size=2, rank=rank)
            op = self._make_dtensor_shard_op(mesh, placement, param_shape=(2, 16, 64), local_shape=(2, 8, 64))

            self.assertEqual(op.shard_tensor(packed).shape, (2, 8, 64))
            self.assertEqual(op.shard_tensor(unpacked_expert, tensor_idx=0).shape, (8, 64))

    def test_packed_rowwise_packed_and_unpacked_shapes(self):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(16, 64)))
        placement = self._get_parameter_placements(module, PackedRowwiseParallel())["weight"]
        packed = torch.randn(16, 64)
        unpacked = torch.randn(16, 32)

        self.assertEqual(placement.dim, -1)
        self.assertEqual(placement.split_factor, 2)
        for rank in range(2):
            mesh = self.MockDeviceMesh(world_size=2, rank=rank)
            op = self._make_dtensor_shard_op(mesh, placement, param_shape=(16, 64), local_shape=(16, 32))

            self.assertEqual(op.shard_tensor(packed).shape, (16, 32))
            self.assertEqual(op.shard_tensor(unpacked).shape, (16, 16))

    def test_grouped_gemm_updates_local_expert_count(self):
        module = torch.nn.Module()
        module.num_experts = 8
        module.register_parameter("weight", torch.nn.Parameter(torch.empty(8, 16, 32)))
        grouped_gemm = ALL_PARALLEL_STYLES["grouped_gemm"]

        placements = self._get_parameter_placements(module, grouped_gemm, self.MockDeviceMesh(world_size=4, rank=0))

        self.assertEqual(placements["weight"].dim, 0)
        self.assertEqual(module.num_experts, 2)

    def test_sharding_does_not_create_unrelated_module_attributes(self):
        styles = (ColwiseParallel(), RowwiseParallel(), ALL_PARALLEL_STYLES["grouped_gemm"])

        for style in styles:
            with self.subTest(style=type(style).__name__):
                module = torch.nn.Module()
                module.random_attr = 123
                module.register_parameter("weight", torch.nn.Parameter(torch.empty(8, 16, 32)))

                self._get_parameter_placements(module, style, self.MockDeviceMesh(world_size=4, rank=0))

                self.assertEqual(module.random_attr, 123)
                self.assertFalse(hasattr(module, "num_experts"))


@is_tensor_parallel_test
class TestReplicatedWithGradAllReduce(TestCasePlus):
    """`replicated_with_grad_allreduce` holds a replicated parameter whose gradient is partial.

    The parameter is the same on every rank, but each rank's forward only sees its own shard of the normalized axis,
    so each rank's gradient covers only part of it and the gradients have to be summed over the mesh.
    """

    class ScaleModule(torch.nn.Module):
        """Stands in for a norm: a replicated parameter applied to this rank's own activations."""

        def __init__(self, size):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(size))

        def forward(self, x):
            return x * self.weight

    def _install(self, module, group="process_group"):
        mesh = MagicMock()
        mesh.get_group.return_value = group
        ALL_PARALLEL_STYLES["replicated_with_grad_allreduce"].install_forward(module, mesh)

    @staticmethod
    @contextlib.contextmanager
    def _doubling_all_reduce():
        """Stands in for a two-rank sum of identical contributions, so the reduction is visible in the gradient."""
        with patch.object(
            tensor_parallel.dist, "all_reduce", side_effect=lambda tensor, group=None: tensor.mul_(2)
        ) as all_reduce:
            yield all_reduce

    def test_parameter_gradient_is_summed_over_the_mesh(self):
        module = self.ScaleModule(3)
        self._install(module)
        x = torch.tensor([1.0, 2.0, 3.0])

        with self._doubling_all_reduce() as all_reduce:
            module(x).sum().backward()

        # The reduction runs once, on this rank's own contribution, and what lands in `param.grad` is the sum.
        self.assertEqual(all_reduce.call_count, 1)
        torch.testing.assert_close(module.weight.grad, 2 * x)

    def test_each_backward_is_reduced_exactly_once(self):
        module = self.ScaleModule(3)
        self._install(module)

        with self._doubling_all_reduce() as all_reduce:
            module(torch.ones(3)).sum().backward()
            module(torch.full((3,), 2.0)).sum().backward()

        # Under gradient accumulation each contribution has to be reduced once, so the accumulated gradient is the
        # reduction of the sum: 2 * (1 + 2). A `register_full_backward_hook` reducing `param.grad` cannot do this, as
        # it fires before `AccumulateGrad`: it would reduce nothing on the first backward and then reduce the value
        # already accumulated on the second, giving 2 * 1 + 2.
        self.assertEqual(all_reduce.call_count, 2)
        torch.testing.assert_close(module.weight.grad, torch.full((3,), 6.0))

    def test_dtensor_parameter_is_reduced_without_a_sharding_strategy(self):
        # `dist.all_reduce` on a DTensor raises `NotImplementedError: Operator c10d.allreduce_.default does not have
        # a sharding strategy registered`, so the collective has to run on the local shard. The parameters are
        # DTensors whenever the model is also wrapped by FSDP.
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, distribute_tensor

        store = torch.distributed.FileStore(tempfile.mktemp(), 1)
        torch.distributed.init_process_group("gloo", store=store, rank=0, world_size=1)
        try:
            mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
            module = self.ScaleModule(3)
            module.register_parameter(
                "weight", torch.nn.Parameter(distribute_tensor(module.weight.detach(), mesh, [Replicate()]))
            )
            self._install(module, group=mesh.get_group())

            x = distribute_tensor(torch.tensor([1.0, 2.0, 3.0]), mesh, [Replicate()])
            # Twice, because reducing `param.grad` only reaches a DTensor from the second backward on: on the first
            # there is no gradient to reduce yet.
            module(x).sum().backward()
            module(x).sum().backward()

            self.assertIsNotNone(module.weight.grad)
            torch.testing.assert_close(module.weight.grad.full_tensor(), torch.tensor([2.0, 4.0, 6.0]))
        finally:
            torch.distributed.destroy_process_group()
