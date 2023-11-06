from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from llm_gym.config.config import ProcessGroupBackendEnum
from llm_gym.env_utils import bfSixteen, has_bfloat_support
from llm_gym.gpt2.gpt2_model import NNModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


class Runner(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @abstractmethod
    def wrap(self, model: NNModel, local_rank: int) -> FSDP:
        raise NotImplementedError()


class FSDPRunner(Runner):
    def __init__(self, process_group_backend: ProcessGroupBackendEnum) -> None:
        dist.init_process_group(process_group_backend.value)

    def run(self):
        dist.barrier()
        dist.destroy_process_group()

    def wrap(self, model: NNModel, local_rank: int) -> FSDP:
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
        torch.cuda.set_device(local_rank)

        if has_bfloat_support():
            mp_policy = bfSixteen
        else:
            mp_policy = None  # defaults to fp32

        # model is on CPU before input to FSDP
        model = FSDP(
            model,
            auto_wrap_policy=None,
            mixed_precision=mp_policy,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
        )
        return model
