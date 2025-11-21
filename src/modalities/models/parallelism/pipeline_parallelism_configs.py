from typing import Annotated

from pydantic import BaseModel, Field

from modalities.config.pydantic_if_types import (
    PydanticDeviceMeshIFType,
    PydanticLossIFType,
    PydanticPipelineStageType,
    PydanticPipelineType,
    PydanticPytorchModuleType,
    PydanticStagesGeneratorType,
)
from modalities.models.parallelism.pipeline_parallelism import PipelineSelectionTypes


class FQNsPerStageGeneratorConfig(BaseModel):  # TODO duplicate
    pass


class StagedPipelineConfig(BaseModel):
    whole_model: PydanticPytorchModuleType
    stages_generator: PydanticStagesGeneratorType
    device_mesh: PydanticDeviceMeshIFType
    local_rank: Annotated[int, Field(strict=True, ge=0)]
    pp_schedule_name: str
    num_layers_per_stage: Annotated[int, Field(strict=True, ge=1)]


class ScheduledPipelineConfig(BaseModel):
    loss_fn: PydanticLossIFType
    pp_schedule_name: str
    batch_size: Annotated[int, Field(strict=True, ge=1)]
    microbatch_size: Annotated[int, Field(strict=True, ge=1)]
    pp_degree: Annotated[int, Field(strict=True, ge=2)]
    pipeline: PydanticPipelineType


class ComponentSelectorFromPipelineConfig(BaseModel):
    pipeline: PydanticPipelineType
    selection_type: PipelineSelectionTypes


class PipelineConfig(BaseModel):
    pp_stages: list[PydanticPipelineStageType] = Field(alias="pp_stage")
    model_parts: list[PydanticPytorchModuleType] = Field(alias="model_part")
    pp_schedule: PydanticPipelineType | None = None
