# ruff: noqa: E402

import os
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv

cwd = Path(__file__).resolve().parent.parent
os.chdir(cwd)
CONFIG_FILE_PATH = cwd / "config" / "qwen_config.yaml"
EXPERIMENTS_ROOT_PATH = cwd / "results" / "debug"


# TODO solve this
def _enable_torchtitan_moe_permute_fallback() -> (
    None
):  # VIBECODATA because of Triton C error with Python headers don't know what that is
    """Avoid Triton JIT build for MoE permute indices on systems without Python dev headers."""
    try:
        import torchtitan.models.moe.kernels as kernels
        import torchtitan.models.moe.utils as moe_utils
    except Exception:
        return

    if getattr(kernels, "_modalities_fallback_enabled", False):
        return

    def _fill_indices_torch(
        tokens_per_expert_group: torch.Tensor,
        start_index_values: torch.Tensor,
        write_offsets: torch.Tensor,
        experts_per_rank: int,
        num_ranks: int,
        max_len: int,
    ) -> torch.Tensor:
        device = tokens_per_expert_group.device
        permuted_indices = torch.full((max_len,), -1, dtype=torch.int32, device=device)

        for e in range(experts_per_rank):
            write_start = int(write_offsets[e].item())
            for r in range(num_ranks):
                i = r * experts_per_rank + e
                start_index = int(start_index_values[i].item())
                length = int(tokens_per_expert_group[i].item())
                if length > 0:
                    end_idx = min(write_start + length, max_len)
                    permuted_indices[write_start:end_idx] = torch.arange(
                        start_index,
                        start_index + (end_idx - write_start),
                        dtype=torch.int32,
                        device=device,
                    )
                write_start += length

        return permuted_indices

    _orig_generate_permute_indices = kernels.generate_permute_indices

    def _generate_permute_indices_no_triton(
        tokens_per_expert_group: torch.Tensor,
        experts_per_rank: int,
        num_ranks: int,
        max_len: int,
        alignment: int,
        use_cpu: bool = False,
    ):
        del use_cpu
        start_index_values = torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
        total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)
        total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)
        m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(torch.int32)
        m_offsets = torch.cumsum(m_sizes, 0)
        write_offsets = m_offsets - m_sizes

        permuted_indices = _fill_indices_torch(
            tokens_per_expert_group=tokens_per_expert_group,
            start_index_values=start_index_values,
            write_offsets=write_offsets,
            experts_per_rank=experts_per_rank,
            num_ranks=num_ranks,
            max_len=max_len,
        )
        return permuted_indices, m_sizes, m_offsets.to(torch.int32)

    kernels.generate_permute_indices = _generate_permute_indices_no_triton
    moe_utils.generate_permute_indices = _generate_permute_indices_no_triton
    setattr(kernels, "_modalities_fallback_enabled", True)
    setattr(kernels, "_modalities_generate_permute_indices_original", _orig_generate_permute_indices)


def debug_ep(model):
    # Stima memoria teorica
    total_params = sum(p.numel() for p in model.parameters())
    ep_params = sum(
        p.numel() for m in model.modules() if getattr(m, "_ep_enabled", False) for p in m.parameters(recurse=False)
    )
    dense_params = total_params - ep_params

    print(f"Params totali: {total_params/1e6:.0f}M")
    print(f"Params EP (non shardati): {ep_params/1e6:.0f}M")
    print(f"Params densi (shardati su dp_shard): {dense_params/1e6:.0f}M")

    rank = dist.get_rank()
    free, total = torch.cuda.mem_get_info()
    print(f"[rank{rank}] Memoria dopo init: {(total-free)/1e9:.1f} GB usati")


def main():
    _enable_torchtitan_moe_permute_fallback()
    EXPERIMENTS_ROOT_PATH.mkdir(parents=True, exist_ok=True)

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        modalities_main = Main(
            config_path=CONFIG_FILE_PATH,
            experiments_root_path=EXPERIMENTS_ROOT_PATH,
        )

        components = cast(
            TrainingComponentsInstantiationModel,
            modalities_main.build_components(components_model_type=TrainingComponentsInstantiationModel),
        )

        # WORKAROUNDS (wip)
        # TODO implement those into moe code
        # 1. some parameters remain on cpu
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        for name, param in components.model_raw.named_parameters():
            if param.device.type == "cpu":
                param.data = param.data.to(device)

        # 2. cast EP params to bf16 — FSDP2 skips them via ignored_params, so they stay
        # fp32 from model init. Cast here to match the MixedPrecisionPolicy applied to
        # dense params (param_dtype=BF_16). Halves EP memory: 29 GB → 14.5 GB at tp=4.
        for mod in components.model_raw.modules():
            if getattr(mod, "_ep_enabled", False):
                for pname, p in list(mod._parameters.items()):
                    if isinstance(p, DTensor) and p.dtype != torch.bfloat16:
                        bf16_local = p.to_local().to(torch.bfloat16)
                        bf16_p = DTensor.from_local(bf16_local, p.device_mesh, p.placements, run_check=False)
                        mod._parameters[pname] = torch.nn.Parameter(bf16_p, requires_grad=p.requires_grad)

        debug_ep(components.model_raw)
        modalities_main.run(components)


if __name__ == "__main__":
    main()
