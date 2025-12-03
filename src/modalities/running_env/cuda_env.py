import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from modalities.config.config import ProcessGroupBackendType


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
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.rdvz_port = rdvz_port
        self._original_env: dict[str, str | None] = {}

    def __enter__(self):
        # Store original values
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
            self._original_env[key] = os.environ.get(key)

        # Set new environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.rdvz_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        # Initialize CUDA environment
        super().__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None):
        # Restore original environment variables
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        super().__exit__(exc_type, exc_val, exc_tb)
