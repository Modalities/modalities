from abc import ABC, abstractmethod
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.gym import Gym

import torch
import torch.distributed as dist
from llm_gym.config.config import ProcessGroupBackendEnum
from llm_gym.env_utils import bfSixteen, has_bfloat_support
from llm_gym.models.gpt2.gpt2_model import Block
from llm_gym.models.model import NNModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools
from torch.optim import Optimizer


class Runner(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @abstractmethod
    def wrap(self, model: NNModel) -> FSDP:
        raise NotImplementedError()


class FSDPRunner(Runner):
    def __init__(self, process_group_backend: ProcessGroupBackendEnum, local_rank: int) -> None:
        dist.init_process_group(process_group_backend.value)
        torch.cuda.set_device(local_rank)


    def run(self, model: NNModel, optimizer: Optimizer, gym: Gym, global_train_batch_id: int, checkpointing: Checkpointing):
        fsdp_model = FSDPRunner.wrap_model(model=model)
        
        gym.run(checkpointing=checkpointing)
        
        
        dist.barrier()
        dist.destroy_process_group()

    @staticmethod
    def wrap_model(model: NNModel) -> FSDP:
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

        if has_bfloat_support():
            mp_policy = bfSixteen
        else:
            mp_policy = None  # defaults to fp32

        transformer_auto_wrapper_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Block,
            },
        )

        # model is on CPU before input to FSDP

        fsdp_model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrapper_policy,
            mixed_precision=mp_policy,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
        )
        return fsdp_model
