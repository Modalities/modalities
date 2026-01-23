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
from modalities.utils.deprecated_alias import add_deprecated_alias


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


@add_deprecated_alias("pp_stages", "pp_stage")
@add_deprecated_alias("model_parts", "model_part")
class PipelineConfig(BaseModel):
    pp_stages: list[PydanticPipelineStageType]
    model_parts: list[PydanticPytorchModuleType]
    pp_schedule: PydanticPipelineType | None = None
