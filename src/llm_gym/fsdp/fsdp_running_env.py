from abc import ABC, abstractmethod
from typing import List, Tuple
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.exceptions import RunningEnvError
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
import torch.nn as nn


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
        process_group_backend: ProcessGroupBackendEnum,
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
        pass # TODO uncomment part below
        # dist.barrier()
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
