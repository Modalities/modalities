import functools
from enum import Enum
from typing import Type

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel, ValidationError, validator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from modalities.config.lookup_types import LookupEnum
from modalities.config.types import ProcessGroupBackendType
from modalities.models.gpt2.gpt2_model import Block
from modalities.running_env.env_utils import MixedPrecisionSettings, has_bfloat_support
from modalities.running_env.running_env import RunningEnv


def parse_enum_by_name(name: str, enum_type: Type[Enum]) -> Enum:
    try:
        return enum_type[name]
    except KeyError:
        raise ValidationError(f"Invalid {enum_type} member name: {name}")


transformer_auto_wrapper_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        Block,
    },
)


class AutoWrapPolicies(Enum):
    TRANSFORMER_AUTO_WRAP_POLICY = transformer_auto_wrapper_policy


class FSDPRunningEnvConfig(BaseModel):
    process_group_backend: ProcessGroupBackendType
    local_rank: int
    mixed_precision_settings: MixedPrecisionSettings
    sharding_strategy: ShardingStrategy
    auto_wrap_policy: AutoWrapPolicies

    @validator("mixed_precision_settings", pre=True, always=True)
    def parse_mixed_precision_setting_by_name(cls, name):
        mixed_precision_settings: MixedPrecisionSettings = parse_enum_by_name(
            name=name, enum_type=MixedPrecisionSettings
        )
        if not has_bfloat_support() and (
            mixed_precision_settings == MixedPrecisionSettings.BF_16
            or mixed_precision_settings == MixedPrecisionSettings.BF_16_WORKING
        ):
            raise ValueError("BF16 not supported in the current environment")
        return mixed_precision_settings

    @validator("sharding_strategy", pre=True, always=True)
    def parse_sharding_strategy_by_name(cls, name):
        return parse_enum_by_name(name=name, enum_type=ShardingStrategy)

    @validator("auto_wrap_policy", pre=True, always=True)
    def parse_auto_wrap_policy_by_name(cls, name):
        return parse_enum_by_name(name=name, enum_type=AutoWrapPolicies)


class FSDPRunningEnv(RunningEnv):
    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
        local_rank: int,
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
        auto_wrap_policy: AutoWrapPolicies,
    ) -> None:
        self.process_group_backend = process_group_backend
        self.local_rank = local_rank
        self.mixed_precision_settings = mixed_precision_settings
        self.sharding_strategy = sharding_strategy
        self.auto_wrap_policy = auto_wrap_policy

    def __enter__(self) -> "RunningEnv":
        dist.init_process_group(self.process_group_backend.value)
        torch.cuda.set_device(self.local_rank)
        return self

    def __exit__(self, type, value, traceback):
        pass  # TODO uncomment part below
        # dist.barrier()    # TODO check for concurrency issues
        # dist.destroy_process_group()

    def wrap_model(self, model: nn.Module, sync_module_states: bool) -> FSDP:
        # model is on CPU before input to FSDP
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=self.auto_wrap_policy.value,
            mixed_precision=self.mixed_precision_settings.value,
            sharding_strategy=self.sharding_strategy,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
        )
        return fsdp_model


class RunningEnvTypes(LookupEnum):
    FSDPRunningEnv = FSDPRunningEnv


class RunningEnvConfig(BaseModel):
    type_hint: RunningEnvTypes
    config: FSDPRunningEnvConfig
