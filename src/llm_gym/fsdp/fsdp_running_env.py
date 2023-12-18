import functools
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from llm_gym.config.lookup_types import LookupEnum
from llm_gym.config.types import ProcessGroupBackendType
from llm_gym.env_utils import bfSixteen, has_bfloat_support
from llm_gym.models.gpt2.gpt2_model import Block


class RunningEnv(ABC, object):
    def __enter__(self) -> "RunningEnv":
        raise NotImplementedError

    def __exit__(self, type, value, traceback):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def wrap_model(model: nn.Module, sync_module_states: bool) -> nn.Module:
        raise NotImplementedError


class FSDPRunningEnv(RunningEnv):
    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
        local_rank: int,
        global_train_batch_id: int = 0,
    ) -> None:
        self.global_train_batch_id = global_train_batch_id
        self.process_group_backend = process_group_backend
        self.local_rank = local_rank

    def __enter__(self) -> "RunningEnv":
        dist.init_process_group(self.process_group_backend.value)
        torch.cuda.set_device(self.local_rank)
        return self

    def __exit__(self, type, value, traceback):
        pass  # TODO uncomment part below
        # dist.barrier()    # TODO check for concurrency issues
        # dist.destroy_process_group()

    @staticmethod
    def wrap_model(model: nn.Module, sync_module_states: bool) -> FSDP:
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
            sync_module_states=sync_module_states,
        )
        return fsdp_model


class RunningEnvTypes(LookupEnum):
    FSDPRunningEnv = FSDPRunningEnv


class FSDPRunningEnvConfig(BaseModel):
    process_group_backend: ProcessGroupBackendType
    local_rank: int


class RunningEnvConfig(BaseModel):
    type_hint: RunningEnvTypes
    config: FSDPRunningEnvConfig
