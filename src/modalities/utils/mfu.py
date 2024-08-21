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
    Get the theoretical peak performance of a single GPU in units FLOPs / s for given GPU type.

    Args:
        precision (torch.dtype): The precision of the GPU.
        gpu_type (str): The type of the GPU.

    Returns:
        (Number, optional): The theoretical peak performance of the GPU if the precision and GPU type are valid,
          otherwise None.
    """

    if precision in [torch.float16, torch.bfloat16] and gpu_type in PEAK_PERFORMANCE:
        return PEAK_PERFORMANCE[gpu_type]
    else:
        return None


def get_theoretical_gpu_peak_performance(model: FSDP, world_size: int) -> Optional[Number]:
    """
    Calculates the accummulated theoretical peak performance based on all GPUs, i.e.,
      #GPU=world_size, in units FLOPs / s for given gpu type.

    Args:
        model (FSDP): The model for which to calculate the theoretical peak performance.
        world_size (int): The number of GPUs used in parallel.

    Returns:
        (Number, optional): The accummulated theoretical peak performance of all GPUs,
          or None if it cannot be calculated.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        precision = model.mixed_precision.param_dtype
        if model.mixed_precision.reduce_dtype != precision or model.mixed_precision.buffer_dtype != precision:
            warnings.warn(f"Could not get theoretical GPU peak performance for mixed precision type = {precision}.")
            return None
        else:
            device_name = torch.cuda.get_device_name()
            if device_name.startswith("NVIDIA A100"):
                single_gpu_peak_performance = _get_theoretical_gpu_peak_performance_single(precision, "A100")
            elif device_name.startswith("NVIDIA H100"):
                single_gpu_peak_performance = _get_theoretical_gpu_peak_performance_single(precision, "H100")
            else:
                warnings.warn(f"Could not get theoretical GPU peak performance for unknown device = {device_name}.")
                return None
            if single_gpu_peak_performance is None:
                warnings.warn(
                    f"Could not get theoretical GPU peak performance for device = {device_name} "
                    f"and mixed precision type = {precision}."
                )
                return None
            else:
                return single_gpu_peak_performance * world_size
    else:
        return None


def get_theoretical_flops_per_token(model: FSDP) -> Tuple[Optional[int], Optional[int]]:
    """
    Calculates the theoretical number of floating point operations (FLOPs) per token for a given model.
    compute theoretical_flops_per_token = 6*N + 12*L*T*H
    See App. B in the PaLM paper (https://arxiv.org/pdf/2204.02311)

    Args:
        model (FSDP): The model for which to calculate the FLOPs per token.

    Returns:
        Tuple[(int, optional), (int, optional)]: A tuple containing the theoretical FLOPs per token
          and the sequence length.
            - Theoretical FLOPs per token: The estimated number of FLOPs required to process each token in the model.
            - Sequence length: The length of the input sequence. Needed to convert samples to tokens in compute_mfu.
            If CUDA is not available, returns (None, None).
    """
    if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0
    ):  # NOTE: This is a workaround to make cpu-only tests work
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
    Computes the Model Flops Utilization (MFU).

    compute mfu = throughput * theoretical_flops_per_token / theoretical_gpu_peak_performance
    units:  [1] = [tokens/s] * [FLOPs / token]             / [FLOPs / s]

    Args:
        num_samples_per_second (torch.Tensor): The number of samples per second.
        sequence_length (int): The length of the sequence.
        theoretical_flops_per_token (Optional[Number]): The theoretical number of floating-point operations per token.
        theoretical_gpu_peak_performance (Optional[Number]): The theoretical peak performance of the GPU.

    Returns:
        torch.Tensor: The computed MFU.

    Note:
        - If either `theoretical_flops_per_token` or `theoretical_gpu_peak_performance` is None,
          the function returns -1.0.
        - The returned value needs to be a float for EvaluationResultBatch.
    """

    if theoretical_flops_per_token is None or theoretical_gpu_peak_performance is None:
        return torch.tensor(-1.0)  # needs to be float for EvaluationResultBatch
    else:
        num_tokens_per_second = num_samples_per_second * sequence_length
        return num_tokens_per_second * theoretical_flops_per_token / theoretical_gpu_peak_performance
