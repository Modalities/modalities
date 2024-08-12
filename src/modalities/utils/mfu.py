from typing import Optional, Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.types import Number

from modalities.util import get_total_number_of_trainable_parameters

PEAK_PERFORMANCE = {
    "A100": 312e12,  # TODO: double-check (also floating point precision types)
    "H100": 989e12,  # TODO: double-check (also floating point precision types)
}


def get_theoretical_gpu_peak_performance(world_size: int) -> Optional[Number]:
    """
    returns theoretical gpu peak performance in units FLOPs / s for given gpu type
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:  # necessary for cpu-only tests
        device_name = torch.cuda.get_device_name()
        if device_name.startswith("NVIDIA A100"):
            return PEAK_PERFORMANCE["A100"] * world_size
        elif device_name.startswith("NVIDIA H100"):
            return PEAK_PERFORMANCE["H100"] * world_size
        else:
            print(
                f"WARNING: could not get theoretical peak performance for found device = {device_name}"
            )  # TODO: print as warning
            return None
    else:
        return None


def get_theoretical_flops_per_token(model: FSDP) -> Tuple[Optional[int], Optional[int]]:
    """
    compute theoretical_flops_per_token = 6*N + 12*L*T*H
    see App. B in the PaLM paper (https://arxiv.org/pdf/2204.02311)

    Returns:
        theoretical_flops_per_token
        sequence_length (needed to convert samples to tokens in compute_mfu)
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:  # necessary for cpu-only tests
        N = get_total_number_of_trainable_parameters(model)
        try:
            L = model.module.n_layer
            T = model.module.sequence_length
            H = model.module.n_embd
            return 6 * N + 12 * L * T * H, T
        except AttributeError:
            return None, None
    else:
        return None, None


def compute_mfu(
    num_samples_per_second: torch.Tensor,
    sequence_length: int,
    theoretical_flops_per_token: Optional[Number],
    theoretical_gpu_peak_performance: Optional[Number],
) -> torch.Tensor:
    """
    compute mfu = throughput * theoretical_flops_per_token / theoretical_gpu_peak_performance
    units:  [1] = [tokens/s] * [FLOPs / token]             / [FLOPs / s]
    """
    if theoretical_flops_per_token is None or theoretical_gpu_peak_performance is None:
        return torch.tensor(-1.0)  # needs to be float for EvaluationResultBatch
    else:
        num_tokens_per_second = num_samples_per_second * sequence_length
        return num_tokens_per_second * theoretical_flops_per_token / theoretical_gpu_peak_performance
