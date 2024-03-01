import os

import torch
import torch.distributed as dist

from modalities.config.config import ProcessGroupBackendType


class CudaEnv:
    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
    ) -> None:
        self.process_group_backend = process_group_backend
        # TODO we might want to set this from outside via the config
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    def __enter__(self) -> "CudaEnv":
        dist.init_process_group(self.process_group_backend.value)
        torch.cuda.set_device(self.local_rank)
        return self

    def __exit__(self, type, value, traceback):
        pass
        # TODO uncomment part below
        # dist.barrier()    # TODO check for concurrency issues
        # dist.destroy_process_group()
