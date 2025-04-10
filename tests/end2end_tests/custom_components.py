import os
from typing import Any, Optional

from pydantic import BaseModel

from modalities.batch import EvaluationResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.logging_broker.messages import Message
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.running_env.cuda_env import CudaEnv


class SaveAllResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def __init__(self):
        self.message_list: list[Message[EvaluationResultBatch]] = []

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        self.message_list.append(message)

    def consume_dict(self, mesasge_dict: dict[str, Any]):
        pass


class SaveAllResultSubscriberConfig(BaseModel):
    pass


class MultiProcessingCudaEnv(CudaEnv):
    """Context manager to set the CUDA environment for distributed training."""

    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
        global_rank: int,
        local_rank: int,
        world_size: int,
        rdvz_port: int,
    ) -> None:
        super().__init__(process_group_backend=process_group_backend)
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.rdvz_port = rdvz_port
        self._original_env: dict[str, Optional[str]] = {}

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original environment variables
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        super().__exit__(exc_type, exc_val, exc_tb)
