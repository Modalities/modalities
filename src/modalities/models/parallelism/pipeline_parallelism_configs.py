from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, model_validator

from modalities.config.pydantic_if_types import (
    PydanticDeviceMeshIFType,
    PydanticLossIFType,
    PydanticPipelineStageType,
    PydanticPipelineType,
    PydanticPytorchModuleType,
    PydanticStagesGeneratorType,
)
from modalities.models.parallelism.pipeline_parallelism import PipelineSelectionTypes
from modalities.running_env.env_utils import PyTorchDtypes
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_parallel_degree
from modalities.utils.deprecated_alias import add_deprecated_alias

# All static pipeline I/O fields; they must be provided together (see StagedPipelineConfig).
STATIC_IO_FIELD_NAMES = (
    "static_io_microbatch_size",
    "static_io_sequence_length",
    "static_io_hidden_dim",
    "static_io_vocab_size",
    "static_io_dtype",
)


def validate_static_io_settings(static_io_settings: dict[str, Any], expert_parallel_degree: int) -> None:
    """Validates the static pipeline I/O metadata of a staged pipeline configuration.

    Args:
        static_io_settings (dict[str, Any]): The values of all STATIC_IO_FIELD_NAMES fields.
        expert_parallel_degree (int): The expert parallelism degree of the run.

    Raises:
        ValueError: If the settings are only partially specified, or if they are absent while expert
            parallelism is used.
    """
    missing = [name for name in STATIC_IO_FIELD_NAMES if static_io_settings.get(name) is None]
    if missing and len(missing) != len(STATIC_IO_FIELD_NAMES):
        raise ValueError(
            "Static pipeline I/O metadata must be specified either completely or not at all, but the "
            f"following fields are missing: {missing}. Provide all of {list(STATIC_IO_FIELD_NAMES)} "
            "or none of them."
        )
    if missing and expert_parallel_degree > 1:
        raise ValueError(
            "Expert parallelism combined with pipeline parallelism requires static pipeline I/O "
            f"metadata, but none was provided. Set all of {list(STATIC_IO_FIELD_NAMES)}. Without "
            "them, the pipeline runs its live shape-inference pass, which deadlocks on the "
            "stage-internal EP all-to-all collectives."
        )


class FQNsPerStageGeneratorConfig(BaseModel):  # TODO duplicate
    pass


class StagedPipelineConfig(BaseModel):
    whole_model: PydanticPytorchModuleType
    stages_generator: PydanticStagesGeneratorType
    device_mesh: PydanticDeviceMeshIFType
    local_rank: Annotated[int, Field(strict=True, ge=0)]
    pp_schedule_name: str
    num_layers_per_stage: Annotated[int, Field(strict=True, ge=1)]
    # Opt-in static pipeline metadata (all of them must be set together, which is enforced by
    # validate_static_io_settings below). Providing these makes the pipeline skip its live
    # shape-inference pass, which is required to compose expert parallelism with pipeline
    # parallelism (the live pass deadlocks on stage-internal EP collectives). Omit them all to keep
    # the default dynamic inference (e.g. for the GPT2 path). `static_io_dtype` must match the
    # dtype of the actual inter-stage activations, i.e. the mixed-precision `param_dtype`.
    static_io_microbatch_size: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    static_io_sequence_length: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    static_io_hidden_dim: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    static_io_vocab_size: Optional[Annotated[int, Field(strict=True, ge=1)]] = None
    static_io_dtype: Optional[PyTorchDtypes] = None

    @model_validator(mode="after")
    def _validate_static_io_settings(self) -> "StagedPipelineConfig":
        """Enforces that static pipeline I/O metadata is either fully specified or fully absent, and
        that it is present whenever expert parallelism is combined with pipeline parallelism.

        A partially specified group would silently fall back to dynamic shape inference, whose live
        forward/backward pass deadlocks against the stage-internal EP all-to-all -- i.e. a single
        typo or omitted field would turn a working EP+PP run into a hang.
        """
        validate_static_io_settings(
            static_io_settings={name: getattr(self, name) for name in STATIC_IO_FIELD_NAMES},
            expert_parallel_degree=get_parallel_degree(self.device_mesh, [ParallelismDegrees.EP]),
        )
        return self


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
