from typing import Annotated, Optional

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
    # Opt-in static pipeline metadata (all four must be set together). Providing these makes the
    # pipeline skip its live shape-inference pass, which is required to compose expert parallelism
    # with pipeline parallelism (the live pass deadlocks on stage-internal EP collectives). Omit
    # them to keep the default dynamic inference (e.g. for the GPT2 path).
    static_io_microbatch_size: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    static_io_sequence_length: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    static_io_hidden_dim: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    static_io_vocab_size: Optional[Annotated[int, Field(strict=True, ge=1)]] = None


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
