import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.types import Number

from modalities.util import get_local_number_of_trainable_parameters
from modalities.utils.typing_utils import FSDPX

# A100: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
# H100: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
#       https://www.nvidia.com/en-us/data-center/h100/
#
# NOTE: These values are valid for fp16 and bf16 only
PEAK_PERFORMANCE = {
    "A100": 312e12,
    "H100": 989e12,
}


class MFUCalculatorABC:
    """
    Interface for calculating the Model Flops Utilization (MFU).
    """

    def compute(self, num_samples_per_second: torch.Tensor) -> torch.Tensor:
        """
        Computes the MFU for the given number of samples per second.

        Args:
            num_samples_per_second (torch.Tensor): The number of samples per second.

        Returns:
            torch.Tensor: The computed MFU.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def _compute_mfu_impl(
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
            theoretical_flops_per_token (Optional[Number]): The theoretical number of
                floating-point operations per token.
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

    @staticmethod
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

    @staticmethod
    def _get_theoretical_gpu_peak_performance(wrapped_model: FSDPX, world_size: int) -> Optional[Number]:
        """
        Calculates the accumulated theoretical peak performance based on all GPUs, i.e.,
        #GPU=world_size, in units FLOPs / s for given gpu type.

        Args:
            model (FSDPX): The model for which to calculate the theoretical peak performance.
            world_size (int): The number of GPUs used in parallel.

        Returns:
            (Number, optional): The accumulated theoretical peak performance of all GPUs,
            or None if it cannot be calculated.
        """
        if isinstance(wrapped_model, FSDP1):
            precision = wrapped_model.mixed_precision.param_dtype
            if (
                wrapped_model.mixed_precision.reduce_dtype != precision
                or wrapped_model.mixed_precision.buffer_dtype != precision
            ):
                warnings.warn(f"Could not get theoretical GPU peak performance for mixed precision type = {precision}.")
                return None
        elif isinstance(wrapped_model, FSDP2):
            warnings.warn("MFU is computed based on the assumption that bf16 precision is used.")
            precision = torch.bfloat16
        else:
            raise TypeError(f"Model should be of type FSDPX, but is {type(wrapped_model)} instead.")

        device_name = torch.cuda.get_device_name()
        if device_name.startswith("NVIDIA A100"):
            single_gpu_peak_performance = MFUCalculatorABC._get_theoretical_gpu_peak_performance_single(
                precision, "A100"
            )
        elif device_name.startswith("NVIDIA H100"):
            single_gpu_peak_performance = MFUCalculatorABC._get_theoretical_gpu_peak_performance_single(
                precision, "H100"
            )
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


class GPT2MFUCalculator(MFUCalculatorABC):
    """
    Class to calculate the Model Flops Utilization (MFU) for a given model.
    """

    def __init__(
        self,
        n_layer: int,
        sequence_length: int,
        n_embd: int,
        world_size: int,
        raw_model: nn.Module,
        wrapped_model: FSDPX,
    ):
        self._num_params = get_local_number_of_trainable_parameters(raw_model)
        self._n_layer = n_layer
        self._sequence_length = sequence_length
        self._n_embd = n_embd
        self._theoretical_flops_per_token = self._get_theoretical_flops_per_token()
        self._theoretical_gpu_peak_performance = MFUCalculatorABC._get_theoretical_gpu_peak_performance(
            wrapped_model, world_size
        )

    def _get_theoretical_flops_per_token(self) -> int:
        return 6 * self._num_params + 12 * self._n_layer * self._sequence_length * self._n_embd

    def compute(self, num_samples_per_second: torch.Tensor) -> torch.Tensor:
        """
        Computes the MFU for the given number of samples per second.

        Args:
            num_samples_per_second (torch.Tensor): The number of samples per second.

        Returns:
            torch.Tensor: The computed MFU.
        """
        return MFUCalculatorABC._compute_mfu_impl(
            num_samples_per_second,
            self._sequence_length,
            self._theoretical_flops_per_token,
            self._theoretical_gpu_peak_performance,
        )
