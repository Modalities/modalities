import os
import traceback
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from modalities.config.config import ProcessGroupBackendType
from modalities.utils.logger_utils import get_logger

logger = get_logger(__name__)


class CudaEnv:
    """Context manager to set the CUDA environment for distributed training."""

    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
        timeout_s: int = 600,
    ) -> None:
        """Initializes the CudaEnv context manager with the process group backend.

        Args:
            process_group_backend (ProcessGroupBackendType): Process group backend to be used for distributed training.
        """
        self.process_group_backend = process_group_backend
        self._timeout_s = timeout_s

    def __enter__(self) -> "CudaEnv":
        """Sets the CUDA environment for distributed training.

        Returns:
            CudaEnv: Instance of the CudaEnv context manager.
        """
        dist.init_process_group(self.process_group_backend.value, timeout=timedelta(seconds=self._timeout_s))
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        if local_rank == -1:
            raise ValueError("LOCAL_RANK environment variable is not set. Please set it before using CudaEnv.")
        torch.cuda.set_device(local_rank)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the CUDA environment for distributed training by destroying the process group.

        Args:
            type (Any):
            value (Any):
            traceback (Any):
        """
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        if exc_type is torch.cuda.OutOfMemoryError:
            logger.error(f"[Rank {local_rank}] CUDA OOM during block, emptying cache.")
            torch.cuda.empty_cache()
        if exc_type is not None:
            logger.error(f"[Rank {local_rank}] Exception of type {exc_type} occurred: {exc_val}")
            tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            logger.error(f"[Rank {local_rank}] Traceback:\n{tb_str}")

        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            logger.error(f"[Rank {local_rank}] Error during process group cleanup: {e}")
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"[Rank {local_rank}] Traceback during cleanup:\n{tb_str}")
