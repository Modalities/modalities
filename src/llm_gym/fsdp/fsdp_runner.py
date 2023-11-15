import functools
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from llm_gym.config.lookup_types import LookupEnum
from llm_gym.config.types import ProcessGroupBackendType
from llm_gym.env_utils import bfSixteen, has_bfloat_support
from llm_gym.models.gpt2.gpt2_model import Block
from llm_gym.models.model import NNModel


class Runner(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @abstractmethod
    def wrap(self, model: NNModel, local_rank: int) -> FSDP:
        raise NotImplementedError()


class FSDPRunner(Runner):
    def __init__(self, process_group_backend: ProcessGroupBackendType) -> None:
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


class RunnerTypes(LookupEnum):
    FSDPRunner = FSDPRunner


class FSDPRunnerConfig(BaseModel):
    process_group_backend: ProcessGroupBackendType


class RunnerConfig(BaseModel):
    type_hint: RunnerTypes
    config: FSDPRunnerConfig
