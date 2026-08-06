from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticDeviceMeshIFType, PydanticFSDP2ModuleType, PydanticPipelineType
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv
from tests.utility import find_free_port

_CONFIG_DIR = Path("tests/fsdp2_parallelization/cp_test_configs")

_PARITY_CONFIGS = {
    "cp_2gpu": {
        "fsdp2_config": _CONFIG_DIR / "fsdp2_config.yaml",
        "cp_config": _CONFIG_DIR / "cp_config.yaml",
        "world_size": 2,
    },
    "cp_tp_4gpu": {
        "fsdp2_config": _CONFIG_DIR / "fsdp2_4gpu_config.yaml",
        "cp_config": _CONFIG_DIR / "cp_tp_config.yaml",
        "world_size": 4,
    },
    "cp_pp_4gpu": {
        "fsdp2_config": _CONFIG_DIR / "fsdp2_4gpu_nope_config.yaml",
        "cp_config": _CONFIG_DIR / "cp_pp_config.yaml",
        "world_size": 4,
    },
    "cp_tp_pp_8gpu": {
        "fsdp2_config": _CONFIG_DIR / "fsdp2_8gpu_nope_config.yaml",
        "cp_config": _CONFIG_DIR / "cp_tp_pp_config.yaml",
        "world_size": 8,
    },
}


class _Components(BaseModel):
    model: PydanticFSDP2ModuleType
    device_mesh: PydanticDeviceMeshIFType


class _PPComponents(BaseModel):
    scheduled_pipeline: PydanticPipelineType
    device_mesh: PydanticDeviceMeshIFType


def _build(config_path: Path, tmp_path: Path) -> _Components:
    return Main(config_path, experiments_root_path=tmp_path).build_components(components_model_type=_Components)


def _build_pp(config_path: Path, tmp_path: Path) -> _PPComponents:
    return Main(config_path, experiments_root_path=tmp_path).build_components(components_model_type=_PPComponents)


def _fixed_input(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def _headtail_input(
    inp: torch.Tensor,
    cp_rank: int,
    cp_degree: int,
) -> tuple[torch.Tensor, int, int]:
    """Return (local_input, head_start, tail_start) for HeadTail-sharded CP.

    The HeadTail load balancer interleaves each CP rank's chunk into a head
    portion (the first 1/(2·cp) of the full sequence) and a tail portion (the
    last 1/(2·cp) of the full sequence, counted from the end), so that every
    rank gets an equal mix of early and late tokens.

    Rank r receives:
      head = inp[:, r * chunk : (r+1) * chunk]
      tail = inp[:, (2*cp-1-r) * chunk : (2*cp-r) * chunk]

    Both portions are concatenated as the rank's local input.

    NOTE: Test configs use RoPE with explicit position_ids passed to the CP model
    so that each rank uses the correct global positions instead of a local 0-based
    arange.  The position_ids are constructed from head_start / tail_start below.
    """
    seq_len = inp.shape[1]
    chunk = seq_len // (2 * cp_degree)

    head_start = cp_rank * chunk
    tail_start = (2 * cp_degree - 1 - cp_rank) * chunk

    local_input = torch.cat(
        [
            inp[:, head_start : head_start + chunk],
            inp[:, tail_start : tail_start + chunk],
        ],
        dim=1,
    )
    return local_input, head_start, tail_start


def _run_cp_logit_match_impl(
    process_id: int,
    fsdp2_config: Path,
    cp_config: Path,
    world_size: int,
    port: int,
    tmp_path: Path,
) -> None:
    """Worker run by mp.spawn: verifies that CP (or CP+TP) logits match the FSDP2 baseline.

    The HeadTail dispatcher in PyTorch's experimental ring-attention API works with
    HeadTail-interleaved input.  Each rank's chunk [head_start, tail_start] maps to
    specific global token indices; the ring correctly gates cross-rank K/V using that
    interleaving.  Feeding sequentially sharded input would apply wrong causal masks
    and produce wrong logits.

    Strategy (no all_gather needed):
      1. Build FSDP2 baseline BEFORE the CP class-level patch is applied.
      2. Run FSDP2 forward on the full sequence → reference logits for every token.
      3. Build CP model (applies the class-level patch).
      4. Construct HeadTail-sharded input for this rank; run CP forward.
      5. Each rank independently compares its local CP logits against the matching
         rows of the FSDP2 reference.
    """
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=world_size,
        rdvz_port=port,
    ):
        vocab_size = 50304
        seq_len = 128
        batch_size = 2
        device = torch.device(f"cuda:{process_id}")

        # Build FSDP2 baseline BEFORE the CP class-level patch is applied.
        torch.manual_seed(42)
        fsdp2 = _build(fsdp2_config, tmp_path)

        inp = _fixed_input(batch_size, seq_len, vocab_size, device)
        out_fsdp2 = fsdp2.model({"input_ids": inp})["logits"].float()

        # Build CP (or CP+TP) model.  This applies the class-level CP patch.
        torch.manual_seed(42)
        cp = _build(cp_config, tmp_path)
        cp_mesh = cp.device_mesh[ParallelismDegrees.CP.value]
        cp_rank = dist.get_rank(cp_mesh.get_group())
        cp_degree = cp_mesh.size()

        chunk = seq_len // (2 * cp_degree)
        input_cp, head_start, tail_start = _headtail_input(inp, cp_rank, cp_degree)
        ref = torch.cat(
            [
                out_fsdp2[:, head_start : head_start + chunk, :],
                out_fsdp2[:, tail_start : tail_start + chunk, :],
            ],
            dim=1,
        )

        # Global position indices for this rank's HeadTail-sharded tokens so RoPE uses
        # the correct frequencies instead of a local 0-based arange.
        position_ids = torch.cat(
            [
                torch.arange(head_start, head_start + chunk, device=device),
                torch.arange(tail_start, tail_start + chunk, device=device),
            ]
        ).unsqueeze(
            0
        )  # (1, 2*chunk)

        out_cp_local = cp.model({"input_ids": input_cp, "position_ids": position_ids})["logits"].float()

        assert out_cp_local.shape == ref.shape, f"Shape mismatch: CP={out_cp_local.shape}, ref={ref.shape}"
        assert torch.allclose(out_cp_local, ref, atol=1e-5, rtol=1e-4), (
            f"Logit mismatch on CP rank {cp_rank}: " f"max abs diff = {(out_cp_local - ref).abs().max().item():.2e}"
        )


def _run_cp_pp_loss_match_impl(
    process_id: int,
    fsdp2_config: Path,
    cp_config: Path,
    world_size: int,
    port: int,
    tmp_path: Path,
) -> None:
    """Worker run by mp.spawn: verifies that CP+PP (or CP+TP+PP) losses match the FSDP2 baseline.

    Strategy:
      1. Build FSDP2 (no CP, no PP) baseline BEFORE the CP class-level patch.
         Both the FSDP2 and CP+PP configs initialize the full model (staged_pipeline
         uses 'initialized_model' as whole_model) so all ranks share identical weights.
      2. Run FSDP2 forward on the full sequence → reference logits for every token.
      3. Build CP+PP model (applies the class-level CP patch).
      4. Construct HeadTail-sharded input/target for this CP rank; run PP schedule forward.
      5. On the last PP stage, compare the CP+PP per-rank loss against the reference loss
         computed from slicing the FSDP2 logits to this CP rank's HeadTail token subset.
    """
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=world_size,
        rdvz_port=port,
    ):
        vocab_size = 50304
        seq_len = 128
        batch_size = 2
        device = torch.device(f"cuda:{process_id}")

        # Build FSDP2 baseline BEFORE the CP class-level patch is applied.
        torch.manual_seed(42)
        fsdp2 = _build(fsdp2_config, tmp_path)

        # Generate a fixed (seq_len+1)-length sequence so we have both input_ids and
        # the shifted target_ids required for the CLM cross-entropy loss.
        torch.manual_seed(0)
        full_seq = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
        inp = full_seq[:, :seq_len]  # (batch, seq_len) — input_ids
        target = full_seq[:, 1:]  # (batch, seq_len) — target_ids

        # Full-sequence forward pass on the FSDP2 baseline.
        with torch.no_grad():
            out_fsdp2 = fsdp2.model({"input_ids": inp})["logits"].float()  # (batch, seq_len, vocab)

        # Build CP+PP (or CP+TP+PP) model.  This applies the class-level CP patch.
        torch.manual_seed(42)
        cp_pp = _build_pp(cp_config, tmp_path)

        cp_mesh = cp_pp.device_mesh[ParallelismDegrees.CP.value]
        cp_rank = dist.get_rank(cp_mesh.get_group())
        cp_degree = cp_mesh.size()

        chunk = seq_len // (2 * cp_degree)
        input_cp, head_start, tail_start = _headtail_input(inp, cp_rank, cp_degree)
        position_ids = (
            torch.cat(
                [
                    torch.arange(head_start, head_start + chunk, device=device),
                    torch.arange(tail_start, tail_start + chunk, device=device),
                ]
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        targets_cp = torch.cat(
            [
                target[:, head_start : head_start + chunk],
                target[:, tail_start : tail_start + chunk],
            ],
            dim=1,
        )  # (batch, 2*chunk)

        # Reference CE loss over this CP rank's token subset, computed from FSDP2 logits.
        ref_logits = torch.cat(
            [
                out_fsdp2[:, head_start : head_start + chunk, :],
                out_fsdp2[:, tail_start : tail_start + chunk, :],
            ],
            dim=1,
        )  # (batch, 2*chunk, vocab)
        ref_loss = F.cross_entropy(
            ref_logits.reshape(-1, vocab_size),
            targets_cp.reshape(-1).long(),
        )

        # Run CP+PP forward (eval = no-grad forward only).
        scheduled_pipeline = cp_pp.scheduled_pipeline
        pp_schedule = scheduled_pipeline.pp_schedule
        targets_pp, losses = (targets_cp.contiguous(), []) if scheduled_pipeline.has_last_pp_stage else (None, None)
        with torch.no_grad():
            if scheduled_pipeline.has_first_pp_stage:
                pp_schedule.eval(input_cp.contiguous(), position_ids, target=targets_pp, losses=losses)
            else:
                pp_schedule.eval(target=targets_pp, losses=losses)

        if scheduled_pipeline.has_last_pp_stage:
            pp_loss = torch.mean(torch.stack(losses)).to(losses[0].device).float()
            assert torch.allclose(pp_loss, ref_loss, atol=1e-5, rtol=1e-4), (
                f"Loss mismatch on CP rank {cp_rank}: "
                f"PP loss = {pp_loss.item():.6f}, ref = {ref_loss.item():.6f}, "
                f"abs diff = {(pp_loss - ref_loss).abs().item():.2e}"
            )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This test requires at least 2 GPUs",
)
class TestContextParallelism:
    def test_cp_output_matches_fsdp2_baseline(self, tmp_path: Path):
        cfg = _PARITY_CONFIGS["cp_2gpu"]
        mp.spawn(
            _run_cp_logit_match_impl,
            args=(cfg["fsdp2_config"], cfg["cp_config"], cfg["world_size"], find_free_port(), tmp_path),
            nprocs=cfg["world_size"],
            join=True,
        )

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4,
        reason="This test requires at least 4 GPUs",
    )
    def test_cp_tp_output_matches_fsdp2_baseline(self, tmp_path: Path):
        cfg = _PARITY_CONFIGS["cp_tp_4gpu"]
        mp.spawn(
            _run_cp_logit_match_impl,
            args=(cfg["fsdp2_config"], cfg["cp_config"], cfg["world_size"], find_free_port(), tmp_path),
            nprocs=cfg["world_size"],
            join=True,
        )

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4,
        reason="This test requires at least 4 GPUs",
    )
    def test_cp_pp_output_matches_fsdp2_baseline(self, tmp_path: Path):
        cfg = _PARITY_CONFIGS["cp_pp_4gpu"]
        mp.spawn(
            _run_cp_pp_loss_match_impl,
            args=(cfg["fsdp2_config"], cfg["cp_config"], cfg["world_size"], find_free_port(), tmp_path),
            nprocs=cfg["world_size"],
            join=True,
        )

    @pytest.mark.skipif(
        torch.cuda.device_count() < 8,
        reason="This test requires at least 8 GPUs",
    )
    def test_cp_tp_pp_output_matches_fsdp2_baseline(self, tmp_path: Path):
        cfg = _PARITY_CONFIGS["cp_tp_pp_8gpu"]
        mp.spawn(
            _run_cp_pp_loss_match_impl,
            args=(cfg["fsdp2_config"], cfg["cp_config"], cfg["world_size"], find_free_port(), tmp_path),
            nprocs=cfg["world_size"],
            join=True,
        )
