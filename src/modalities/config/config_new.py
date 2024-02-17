from pathlib import Path
from typing import Annotated, Any

import torch.nn as nn
from pydantic import BaseModel, GetCoreSchemaHandler, PositiveInt, conint
from pydantic_core import core_schema

from modalities.checkpointing.checkpointing_execution import CheckpointingExecutionIF
from modalities.checkpointing.checkpointing_strategies import CheckpointingStrategyIF
from modalities.config.lookup_types import LookupEnum
from modalities.running_env.running_env import RunningEnv


class PydanticRunningEnvIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(RunningEnv),
            python_schema=core_schema.is_instance_schema(RunningEnv),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticCheckpointingStrategyIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(CheckpointingStrategyIF),
            python_schema=core_schema.is_instance_schema(CheckpointingStrategyIF),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticCheckpointingExecutionIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(CheckpointingExecutionIF),
            python_schema=core_schema.is_instance_schema(CheckpointingExecutionIF),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticModelIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(nn.Module),
            python_schema=core_schema.is_instance_schema(nn.Module),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


PydanticRunningEnvType = Annotated[RunningEnv, PydanticRunningEnvIF]
PydanticCheckpointingStrategyIFType = Annotated[CheckpointingStrategyIF, PydanticCheckpointingStrategyIF]
PydanticCheckpointingExecutionIFType = Annotated[CheckpointingExecutionIF, PydanticCheckpointingExecutionIF]
PydanticModelIFType = Annotated[nn.Module, PydanticModelIF]


class PassType(LookupEnum):
    BY_VALUE = "by_value"
    BY_REFERENCE = "by_reference"


class ReferenceConfig(BaseModel):
    instance_key: str
    pass_type: PassType


class CLMCrossEntropyLossConfig(BaseModel):
    target_key: str
    prediction_key: str


# Checkpointing


class SaveEveryKStepsCheckpointingStrategyConfig(BaseModel):
    k: PositiveInt


class SaveKMostRecentCheckpointsStrategyConfig(BaseModel):
    k: conint(ge=-1)


class FSDPToDiscCheckpointingConfig(BaseModel):
    checkpoint_path: Path
    global_rank: conint(ge=0)
    experiment_id: str
    running_env: PydanticRunningEnvType


class CheckpointingConfig(BaseModel):
    checkpointing_strategy: PydanticCheckpointingStrategyIFType
    checkpointing_execution: PydanticCheckpointingExecutionIFType


class AdamWOptimizerConfig(BaseModel):
    lr: float
    model: PydanticModelIFType
