import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from modalities.config.config import ProcessGroupBackendType
from modalities.utils.env import EnvOverride


class CudaEnv:
    """Context manager to set the CUDA environment for distributed training."""

    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
        timeout_s: int = 600,
        **process_group_kwargs: Any,
    ) -> None:
        """Initializes the CudaEnv context manager with the process group backend.

        Args:
            process_group_backend (ProcessGroupBackendType): Process group backend to be used for distributed training.
            timeout_s (int, optional): Timeout in seconds for process group initialization. Defaults to 600.
            **process_group_kwargs: Additional keyword arguments for process group initialization.
        """
        self.process_group_backend = process_group_backend
        self._timeout_s = timeout_s
        self._process_group_kwargs = process_group_kwargs

    def __enter__(self) -> "CudaEnv":
        """Sets the CUDA environment for distributed training.

        Returns:
            CudaEnv: Instance of the CudaEnv context manager.
        """
        dist.init_process_group(
            self.process_group_backend.value, timeout=timedelta(seconds=self._timeout_s), **self._process_group_kwargs
        )
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
            print(f"[Rank {local_rank}] CUDA OOM during block, emptying cache.")
            torch.cuda.empty_cache()

        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            print(f"[Rank {local_rank}] Error during process group cleanup: {e}")


class MultiProcessingCudaEnv(CudaEnv):
    """Context manager to set the CUDA environment for distributed training."""

    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
        global_rank: int,
        local_rank: int,
        world_size: int,
        rdvz_port: int,
        timeout_s: int = 600,
        **process_group_kwargs: Any,
    ) -> None:
        super().__init__(process_group_backend=process_group_backend, timeout_s=timeout_s, **process_group_kwargs)
        self._env_override = EnvOverride(
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(rdvz_port),
                "RANK": str(global_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
            }
        )

    def __enter__(self) -> "MultiProcessingCudaEnv":
        # Set environment overrides
        self._env_override.__enter__()
        # Initialize CUDA environment
        super().__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None):
        # Restore original environment variables
        self._env_override.__exit__(exc_type, exc_val, exc_tb)
        super().__exit__(exc_type, exc_val, exc_tb)
