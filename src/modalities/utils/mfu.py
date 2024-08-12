import warnings
from typing import Optional, Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.types import Number

from modalities.util import get_total_number_of_trainable_parameters

# A100: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
# H100: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
#       https://www.nvidia.com/en-us/data-center/h100/
#
# NOTE: These values are valid for fp16 and bf16 only
PEAK_PERFORMANCE = {
    "A100": 312e12,
    "H100": 989e12,
}


def _get_theoretical_gpu_peak_performance_single(precision: torch.dtype, gpu_type: str) -> Optional[Number]:
    """
    returns theoretical gpu peak performance for #GPU=1 in units FLOPs / s for given gpu type
    """
    if precision in [torch.float16, torch.bfloat16] and gpu_type in PEAK_PERFORMANCE.keys():
        return PEAK_PERFORMANCE[gpu_type]
    else:
        return None


def get_theoretical_gpu_peak_performance(model: FSDP, world_size: int) -> Optional[Number]:
    """
    returns theoretical gpu peak performance for #GPU=world_size in units FLOPs / s for given gpu type
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:  # necessary for cpu-only tests
        precision = model.mixed_precision.param_dtype
        if model.mixed_precision.reduce_dtype != precision or model.mixed_precision.buffer_dtype != precision:
            warnings.warn("could not get theoretical gpu peak performance for given mixed precision type")
            return None
        else:
            device_name = torch.cuda.get_device_name()
            if device_name.startswith("NVIDIA A100"):
                single_gpu_peak_performance = _get_theoretical_gpu_peak_performance_single(precision, "A100")
            elif device_name.startswith("NVIDIA H100"):
                single_gpu_peak_performance = _get_theoretical_gpu_peak_performance_single(precision, "H100")
            else:
                warnings.warn(f"could not get theoretical gpu peak performance for unknown device = {device_name}")
                return None
            if single_gpu_peak_performance is None:
                warnings.warn(f"could not get theoretical gpu peak performance for {device_name} and {precision}")
                return None
            else:
                return single_gpu_peak_performance * world_size
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
