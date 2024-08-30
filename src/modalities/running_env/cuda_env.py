import os
from typing import Any

import torch
import torch.distributed as dist

from modalities.config.config import ProcessGroupBackendType


class CudaEnv:
    """Context manager to set the CUDA environment for distributed training."""

    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
    ) -> None:
        """Initializes the CudaEnv context manager with the process group backend.

        Args:
            process_group_backend (ProcessGroupBackendType): Process group backend to be used for distributed training.
        """
        self.process_group_backend = process_group_backend
        # TODO we might want to set this from outside via the config
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    def __enter__(self) -> "CudaEnv":
        """Sets the CUDA environment for distributed training.

        Returns:
            CudaEnv: Instance of the CudaEnv context manager.
        """
        dist.init_process_group(self.process_group_backend.value)
        torch.cuda.set_device(self.local_rank)
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any):
        """Exits the CUDA environment for distributed training by destroying the process group.

        Args:
            type (Any):
            value (Any):
            traceback (Any):
        """
        # TODO and NOTE:
        # when we call barrier here and one of the ranks fails, we get stuck here.
        # In the future, we should probably add a timeout here and handle the case when one of the ranks fails.
        # dist.barrier()
        dist.destroy_process_group()
