"""Distributed tests for expert parallelism.

Three things are checked, each of which only manifests with a real process group:

* the routed expert weights end up as DTensors on the ``(dp_shard_mod_ep, ep)`` mesh, holding only
  this rank's share of the experts, while the router and shared experts stay data-parallel,
* a full forward/backward/optimizer step runs, including gradient clipping, which has to combine
  gradient norms across two different device meshes,
* an expert-parallel MoE layer computes exactly what a replicated one computes -- the dispatch and
  combine all-to-all pair must be an identity on the routing result.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import BaseModel
from torch.distributed.tensor import DTensor

from modalities.__main__ import Main
from modalities.batch import InferenceResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import (
    PydanticDeviceMeshIFType,
    PydanticFSDP2ModuleType,
    PydanticLossIFType,
    PydanticOptimizerIFType,
)
from modalities.models.components.moe.experts import ExpertsBackend, GroupedExperts
from modalities.models.components.moe.moe import MoE
from modalities.models.components.moe.router import TopKRouter
from modalities.models.parallelism.expert_parallelism import ExpertParallelGroupedExperts, shard_experts_over_ep_mesh
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_device_mesh
from modalities.training.gradient_clipping.fsdp_gradient_clipper import FSDP2GradientClipper, GradientClippingMode
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv

CONFIG_PATH = Path(os.path.dirname(__file__)) / "nemotron_ep_fsdp2_config.yaml"
WORLD_SIZE = 2
VOCAB_SIZE = 512
SEQ_LEN = 32
NUM_EXPERTS = 8  # must match the config


class _Components(BaseModel):
    initialized_model: PydanticFSDP2ModuleType
    device_mesh: PydanticDeviceMeshIFType
    optimizer: PydanticOptimizerIFType
    loss_fn: PydanticLossIFType


def _build_components(tmp_path: Path) -> _Components:
    main_obj = Main(CONFIG_PATH, experiments_root_path=tmp_path)
    return main_obj.build_components(components_model_type=_Components)


def _sharding_worker(process_id: int, tmp_path: str, rdvz_port: int):
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=WORLD_SIZE,
        rdvz_port=rdvz_port,
    ):
        components = _build_components(Path(tmp_path))
        model = components.initialized_model
        device_mesh = components.device_mesh

        # The mesh must expose the EP dimensions and still resolve `dp_shard`, which is now a
        # flattened alias rather than a named dimension.
        assert device_mesh[ParallelismDegrees.EP.value].size() == WORLD_SIZE
        assert device_mesh[ParallelismDegrees.DP_SHARD.value].size() == WORLD_SIZE
        assert device_mesh[ParallelismDegrees.DP_SHARD_MOD_EP.value].size() == 1

        parameters = dict(model.named_parameters())

        def find(suffix: str) -> torch.Tensor:
            matches = [param for name, param in parameters.items() if name.endswith(suffix)]
            assert matches, f"no parameter ending in {suffix!r}; available: {sorted(parameters)}"
            return matches[0]

        # Expert weights: global shape unchanged, but the local shard holds only this rank's experts
        # and lives on the two-dimensional expert mesh.
        expert_weight = find("experts.experts.w1")
        assert isinstance(expert_weight, DTensor)
        assert expert_weight.shape[0] == NUM_EXPERTS, expert_weight.shape
        assert expert_weight.device_mesh.mesh_dim_names == (
            ParallelismDegrees.DP_SHARD_MOD_EP.value,
            ParallelismDegrees.EP.value,
        )
        local_expert_weight = expert_weight.to_local()
        assert local_expert_weight.shape[0] == NUM_EXPERTS // WORLD_SIZE, local_expert_weight.shape

        # The router is not expert-parallel: it stays on the plain data-parallel mesh.
        gate = find("moe.router.gate.weight")
        assert gate.device_mesh.mesh_dim_names == (ParallelismDegrees.DP_SHARD.value,)

        # Everything must be materialized off the meta device and finite.
        for name, param in parameters.items():
            local = param.to_local() if isinstance(param, DTensor) else param
            assert local.device.type != "meta", f"{name} is still on the meta device"
            if local.numel() > 0:
                assert torch.isfinite(local).all(), f"{name} is not finite"


def _training_step_worker(process_id: int, tmp_path: str, rdvz_port: int):
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=WORLD_SIZE,
        rdvz_port=rdvz_port,
    ):
        components = _build_components(Path(tmp_path))
        model = components.initialized_model
        optimizer = components.optimizer
        loss_fn = components.loss_fn
        # Gradient clipping has to reduce norms over both the data-parallel and the expert mesh.
        clipper = FSDP2GradientClipper(
            model_parts=model, max_norm=1.0, norm_type=GradientClippingMode.P2_NORM, device_mesh=components.device_mesh
        )

        # Different data per rank, so the dispatch all-to-all actually moves tokens.
        generator = torch.Generator(device="cuda").manual_seed(process_id)
        inputs = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device="cuda", generator=generator)
        targets = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device="cuda", generator=generator)

        losses = []
        for _ in range(2):
            predictions = model({"input_ids": inputs})
            batch = InferenceResultBatch(targets={"target_ids": targets}, predictions=predictions)
            loss = loss_fn(batch)
            assert torch.isfinite(loss), loss
            loss.backward()
            grad_norm = clipper.clip_gradients()
            assert torch.isfinite(grad_norm), grad_norm
            assert grad_norm > 0, grad_norm
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        # Ranks must agree on the loss trend having moved; the values themselves differ by data.
        assert losses[1] != losses[0]

        # Auxiliary-loss-free load balancing must still work under expert parallelism. The router
        # counts tokens per *global* expert before dispatch, and the counts are reduced over every
        # rank that holds a distinct data shard -- which, with expert parallelism carved out of
        # dp_shard, is all of them. A bias stuck at zero would mean the reduction group is wrong.
        moe_layers = [module for name, module in model.named_modules() if name.endswith(".moe")]
        assert moe_layers, "no MoE layer found"
        for moe in moe_layers:
            expert_bias = moe.router.expert_bias
            assert expert_bias.shape == (NUM_EXPERTS,), expert_bias.shape
            assert not torch.all(expert_bias == 0), "expert bias was never updated"
            # The counters are reset by the hook after each step.
            assert torch.all(moe.router.tokens_per_expert == 0), moe.router.tokens_per_expert
            # Every rank must arrive at the same bias, since the counts were all-reduced.
            gathered = [torch.zeros_like(expert_bias) for _ in range(WORLD_SIZE)]
            dist.all_gather(gathered, expert_bias)
            torch.testing.assert_close(gathered[0], gathered[-1], rtol=0, atol=0)


def _equivalence_worker(process_id: int, tmp_path: str, rdvz_port: int):
    """An expert-parallel MoE must match a replicated MoE bit-for-bit in the forward pass."""
    del tmp_path
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=WORLD_SIZE,
        rdvz_port=rdvz_port,
    ):
        n_embd, ffn_hidden, top_k = 64, 128, 2
        device_mesh = get_device_mesh(
            device_type="cuda",
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=WORLD_SIZE,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            enable_loss_parallel=False,
            world_size=WORLD_SIZE,
            expert_parallel_degree=WORLD_SIZE,
        )
        ep_mesh = device_mesh[ParallelismDegrees.EP.value]

        def build() -> MoE:
            torch.manual_seed(1234)  # identical on every rank
            return (
                MoE(
                    router=TopKRouter(n_embd=n_embd, num_experts=NUM_EXPERTS, top_k=top_k, route_scale=1.0),
                    experts=GroupedExperts(
                        n_embd=n_embd,
                        ffn_hidden=ffn_hidden,
                        num_experts=NUM_EXPERTS,
                        backend=ExpertsBackend.GROUPED_MM,
                    ),
                    shared_experts=None,
                    aux_loss_coeff=0.0,
                )
                .cuda()
                .to(torch.bfloat16)
            )

        replicated = build()
        reference_w1 = replicated.experts.w1.detach().clone()
        reference_w2 = replicated.experts.w2.detach().clone()

        expert_parallel = build()
        shard_experts_over_ep_mesh(expert_parallel.experts, ep_mesh)
        local_experts = NUM_EXPERTS // WORLD_SIZE
        lo, hi = process_id * local_experts, (process_id + 1) * local_experts
        with torch.no_grad():
            expert_parallel.experts.w1.to_local().copy_(reference_w1[lo:hi])
            expert_parallel.experts.w2.to_local().copy_(reference_w2[lo:hi])
        expert_parallel.experts = ExpertParallelGroupedExperts(experts=expert_parallel.experts, ep_mesh=ep_mesh)

        # Per-rank distinct input so routing differs and the all-to-all is non-trivial.
        torch.manual_seed(999 + process_id)
        x = torch.randn(2, 64, n_embd, device="cuda", dtype=torch.bfloat16)

        torch.testing.assert_close(expert_parallel(x), replicated(x), rtol=0, atol=0)

        # Gradients must match too, up to bf16 accumulation order. The replicated run only sees this
        # rank's tokens, so its expert gradients are summed over ranks before comparing against the
        # expert-parallel run, whose local experts saw every rank's tokens.
        replicated(x).float().pow(2).mean().backward()
        expert_parallel(x).float().pow(2).mean().backward()
        reference_grad = replicated.experts.w1.grad.float()
        dist.all_reduce(reference_grad)
        local_grad = expert_parallel.experts.experts.w1.grad
        local_grad = (local_grad.to_local() if isinstance(local_grad, DTensor) else local_grad).float()
        torch.testing.assert_close(local_grad, reference_grad[lo:hi], rtol=1e-2, atol=1e-4)


@pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE, reason=f"expert parallelism test requires {WORLD_SIZE} GPUs"
)
@pytest.mark.parametrize(
    "worker, rdvz_port",
    [(_sharding_worker, 22431), (_training_step_worker, 22432), (_equivalence_worker, 22433)],
    ids=["sharding", "training_step", "equivalence_to_replicated"],
)
def test_expert_parallelism(worker, rdvz_port, tmp_path):
    mp.spawn(worker, args=(str(tmp_path), rdvz_port), nprocs=WORLD_SIZE, join=True)
