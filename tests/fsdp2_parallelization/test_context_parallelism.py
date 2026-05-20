from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticDeviceMeshIFType, PydanticFSDP2ModuleType
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv
from tests.utility import find_free_port

_FSDP2_2GPU = Path("tests/fsdp2_parallelization/cp_test_configs/fsdp2_config.yaml")
_CP_2GPU = Path("tests/fsdp2_parallelization/cp_test_configs/cp_config.yaml")
_FSDP2_4GPU = Path("tests/fsdp2_parallelization/cp_test_configs/fsdp2_4gpu_config.yaml")
_CP_TP_4GPU = Path("tests/fsdp2_parallelization/cp_test_configs/cp_tp_config.yaml")


class _Components(BaseModel):
    model: PydanticFSDP2ModuleType
    device_mesh: PydanticDeviceMeshIFType


def _build(config_path: Path, tmp_path: Path) -> _Components:
    return Main(config_path, experiments_root_path=tmp_path).build_components(components_model_type=_Components)


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

    NOTE: Test configs deliberately omit RoPE (qkv_transforms: []).
    RotaryTransform builds position indices from local sequence length
    (torch.arange(local_seq_len)), so CP ranks > 0 would get local positions
    [0..local_seq-1] instead of their correct global positions.  Stripping
    RoPE isolates the ring-attention communication test from that known
    positional-encoding limitation.
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


def _run_cp_parity_impl(
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

        out_cp_local = cp.model({"input_ids": input_cp})["logits"].float()

        assert out_cp_local.shape == ref.shape, f"Shape mismatch: CP={out_cp_local.shape}, ref={ref.shape}"
        assert torch.allclose(out_cp_local, ref, atol=1e-5, rtol=1e-4), (
            f"Logit mismatch on CP rank {cp_rank}: " f"max abs diff = {(out_cp_local - ref).abs().max().item():.2e}"
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This test requires at least 2 GPUs",
)
class TestContextParallelism:
    def test_cp_output_matches_fsdp2_baseline(self, tmp_path: Path):
        mp.spawn(
            _run_cp_parity_impl,
            args=(_FSDP2_2GPU, _CP_2GPU, 2, find_free_port(), tmp_path),
            nprocs=2,
            join=True,
        )

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4,
        reason="This test requires at least 4 GPUs",
    )
    def test_cp_tp_output_matches_fsdp2_baseline(self, tmp_path: Path):
        mp.spawn(
            _run_cp_parity_impl,
            args=(_FSDP2_4GPU, _CP_TP_4GPU, 4, find_free_port(), tmp_path),
            nprocs=4,
            join=True,
        )
