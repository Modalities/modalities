"""Distributed FSDP2 tests for the Nemotron hybrid Mamba-Transformer.

These cover the parts that only manifest under real sharding:

* the mixture-of-experts weights are sharded along the expert dimension,
* the state space parameters survive meta-device materialization and initialization,
* a full forward/backward/optimizer step runs, including the auxiliary-loss-free expert bias
  update, which reduces token counts across data-parallel ranks.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.batch import InferenceResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import (
    PydanticDeviceMeshIFType,
    PydanticFSDP2ModuleType,
    PydanticLossIFType,
    PydanticOptimizerIFType,
)
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv

CONFIG_PATH = Path(os.path.dirname(__file__)) / "nemotron_fsdp2_config.yaml"
WORLD_SIZE = 2
VOCAB_SIZE = 512
SEQ_LEN = 32


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

        # Activation checkpointing inserts wrapper modules into the fully qualified names, so match
        # on the suffix rather than on an exact path.
        parameters = dict(model.named_parameters())

        def find(suffix: str) -> torch.Tensor:
            matches = [param for name, param in parameters.items() if name.endswith(suffix)]
            assert matches, f"no parameter ending in {suffix!r}; available: {sorted(parameters)}"
            return matches[0]

        # Expert weights are stored as (num_experts, ffn_hidden, n_embd); FSDP2 shards dim 0, so
        # each rank owns a slice of the experts.
        expert_weight = find("moe.experts.w1")
        assert expert_weight.shape == (8, 64, 256), expert_weight.shape
        local_expert_weight = expert_weight.to_local()
        assert local_expert_weight.shape[0] == 8 // WORLD_SIZE, local_expert_weight.shape

        # Every parameter must be materialized and finite after initialization.
        for name, param in parameters.items():
            local = param.to_local() if hasattr(param, "to_local") else param
            assert local.device.type != "meta", f"{name} is still on the meta device"
            if local.numel() > 0:
                assert torch.isfinite(local).all(), f"{name} is not finite"

        # The state space parameters must have kept their own distributions, not the normal one.
        A_log = find("mixer.A_log")
        A = torch.exp(A_log.full_tensor() if hasattr(A_log, "full_tensor") else A_log)
        assert A.min() >= 1.0 and A.max() <= 16.0, (A.min().item(), A.max().item())

        D = find("mixer.D")
        D_full = D.full_tensor() if hasattr(D, "full_tensor") else D
        torch.testing.assert_close(D_full, torch.ones_like(D_full))

        # The router gate, by contrast, must have been touched by the normal-distribution
        # initializer, confirming the two initialization paths coexist correctly.
        gate = find("moe.router.gate.weight")
        gate_full = gate.full_tensor() if hasattr(gate, "full_tensor") else gate
        assert gate_full.std().item() == pytest.approx(0.02, rel=0.3)


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

        # Different data per rank, so that the expert-load reduction has something to combine.
        generator = torch.Generator(device="cuda").manual_seed(process_id)
        inputs = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device="cuda", generator=generator)
        targets = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device="cuda", generator=generator)

        losses = []
        for _ in range(2):
            predictions = model({"input_ids": inputs})
            assert "moe_aux_loss" in predictions
            batch = InferenceResultBatch(targets={"target_ids": targets}, predictions=predictions)
            loss = loss_fn(batch)
            assert torch.isfinite(loss), loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        # The load-balancing hook must have moved the expert biases away from zero and reset the
        # token counters. Both are rank-consistent because the counts are all-reduced.
        moe_layers = [module for name, module in model.named_modules() if name.endswith(".moe")]
        assert moe_layers, "no MoE layers found"
        for moe in moe_layers:
            expert_bias = moe.router.expert_bias
            assert expert_bias.abs().max() > 0, "expert bias was never updated"
            assert moe.router.tokens_per_expert.sum() == 0, "token counters were not reset"

        assert all(loss == loss for loss in losses)  # no NaNs


@pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason=f"This test requires {WORLD_SIZE} GPUs")
def test_nemotron_fsdp2_shards_experts_and_initializes_state_space_parameters(tmp_path):
    mp.spawn(_sharding_worker, args=(str(tmp_path), 22421), nprocs=WORLD_SIZE, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason=f"This test requires {WORLD_SIZE} GPUs")
def test_nemotron_fsdp2_training_step_with_load_balancing(tmp_path):
    mp.spawn(_training_step_worker, args=(str(tmp_path), 22422), nprocs=WORLD_SIZE, join=True)
