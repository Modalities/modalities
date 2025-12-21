from pathlib import Path

from pydantic import BaseModel

from modalities.config.lookup_enum import LookupEnum
from modalities.config.pydantic_if_types import PydanticSteppableProfilerIFType


class ModalitiesProfilerActivity(LookupEnum):
    CPU = "CPU"
    CUDA = "CUDA"


class SteppableKernelProfilerConfig(BaseModel):
    """Settings for the kernel profiler."""

    num_wait_steps: int
    num_warmup_steps: int
    num_active_steps: int
    profiler_activities: list[ModalitiesProfilerActivity]
    record_shapes: bool
    with_flops: bool
    with_stack: bool
    with_modules: bool
    output_folder_path: Path
    tracked_ranks: list[int] | None = None


class SteppableMemoryProfilerConfig(BaseModel):
    """Settings for the memory profiler."""

    memory_snapshot_folder_path: Path
    num_wait_steps: int
    num_warmup_steps: int
    num_active_steps: int
    tracked_ranks: list[int] | None = None


class SteppableNoProfilerConfig(BaseModel):
    """Settings for no profiler."""

    pass


class SteppableCombinedProfilerConfig(BaseModel):
    """Settings for combined profilers."""

    profilers: list[PydanticSteppableProfilerIFType]
