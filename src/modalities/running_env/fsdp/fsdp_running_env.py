from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel, validator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from modalities.config.lookup_types import LookupEnum
from modalities.config.types import ProcessGroupBackendType
from modalities.running_env.env_utils import MixedPrecisionSettings, has_bfloat_support
from modalities.running_env.fsdp.fsdp_auto_wrapper import FSDPTransformerAutoWrapPolicyFactory
from modalities.running_env.running_env import RunningEnv
from modalities.util import parse_enum_by_name


class FSDPRunningEnv(RunningEnv):
    def __init__(
        self,
        process_group_backend: ProcessGroupBackendType,
        local_rank: int,
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
        block_names: List[str],
    ) -> None:
        self.process_group_backend = process_group_backend
        self.local_rank = local_rank
        self.mixed_precision_settings = mixed_precision_settings
        self.sharding_strategy = sharding_strategy
        self.block_names = block_names

    def __enter__(self) -> "RunningEnv":
        dist.init_process_group(self.process_group_backend.value)
        torch.cuda.set_device(self.local_rank)
        return self

    def __exit__(self, type, value, traceback):
        pass  # TODO uncomment part below
        # dist.barrier()    # TODO check for concurrency issues
        # dist.destroy_process_group()

    def wrap_model(self, model: nn.Module, sync_module_states: bool) -> FSDP:
        # Here, FSDPTransformerAutoWrapPolicyFactory is hardcoded and should be passed in instead!
        # we also might want to have different auto wrap policies later...
        fsdp_auto_wrap_factory = FSDPTransformerAutoWrapPolicyFactory(model=model, block_names=self.block_names)

        # model is on CPU before input to FSDP
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=fsdp_auto_wrap_factory.get_auto_wrap_policy(),
            mixed_precision=self.mixed_precision_settings.value,
            sharding_strategy=self.sharding_strategy,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
        )
        return fsdp_model


class FSDPRunningEnvConfig(BaseModel):
    process_group_backend: ProcessGroupBackendType
    local_rank: int
    mixed_precision_settings: MixedPrecisionSettings
    sharding_strategy: ShardingStrategy
    block_names: List[str]

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


class RunningEnvTypes(LookupEnum):
    FSDPRunningEnv = FSDPRunningEnv


class RunningEnvConfig(BaseModel):
    type_hint: RunningEnvTypes
    config: FSDPRunningEnvConfig
