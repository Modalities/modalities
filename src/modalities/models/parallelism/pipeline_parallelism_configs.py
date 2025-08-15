from typing import Annotated

from pydantic import BaseModel, Field

from modalities.config.pydantic_if_types import (
    PydanticDeviceMeshIFType,
    PydanticPytorchModuleType,
    PydanticStagesGeneratorType,
)


class FQNsPerStageGeneratorConfig(BaseModel):
    pass


class PipelinedModelConfig(BaseModel):
    whole_model: PydanticPytorchModuleType
    stages_generator: PydanticStagesGeneratorType
    device_mesh: PydanticDeviceMeshIFType
    local_rank: Annotated[int, Field(strict=True, ge=0)]
    pp_schedule_name: str
    num_layers_per_stage: Annotated[int, Field(strict=True, ge=1)]
