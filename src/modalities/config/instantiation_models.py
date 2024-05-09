import os
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union

from pydantic import BaseModel, Field, FilePath, field_validator

from modalities.config.pydanctic_if_types import (
    PydanticCheckpointSavingIFType,
    PydanticGradientClipperIFType,
    PydanticLLMDataLoaderIFType,
    PydanticLossIFType,
    PydanticLRSchedulerIFType,
    PydanticMessageSubscriberIFType,
    PydanticOptimizerIFType,
    PydanticPytorchDeviceType,
    PydanticPytorchModuleType,
    PydanticTextInferenceComponentType,
    PydanticTokenizerIFType,
)
from modalities.config.utils import parse_torch_device


class CudaEnvSettings(BaseModel):
    local_rank: Annotated[int, Field(strict=True, ge=0)]
    world_size: Annotated[int, Field(strict=True, ge=1)]
    global_rank: Annotated[int, Field(strict=True, ge=0)]


class TrainingComponentsInstantiationModel(BaseModel):
    class TrainingSettings(BaseModel):
        class Training(BaseModel):
            global_training_log_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            global_checkpointing_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            global_evaluation_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            do_apply_activation_checkpointing: bool
            gradient_acc_steps: Annotated[int, Field(strict=True, ge=1)]
            local_train_micro_batch_size: Annotated[int, Field(strict=True, ge=1)]
            sequence_length: Annotated[int, Field(strict=True, ge=1)]

        class Paths(BaseModel):
            checkpointing_path: Path

        experiment_id: str
        referencing_keys: Dict[str, str]
        training: Training
        cuda_env: CudaEnvSettings
        paths: Paths

    wrapped_model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType
    scheduler: PydanticLRSchedulerIFType
    loss_fn: Union[PydanticLossIFType, List[PydanticLossIFType]]
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: List[PydanticLLMDataLoaderIFType]
    batch_progress_subscriber: PydanticMessageSubscriberIFType
    evaluation_subscriber: PydanticMessageSubscriberIFType
    checkpoint_saving: PydanticCheckpointSavingIFType
    gradient_clipper: PydanticGradientClipperIFType
    settings: TrainingSettings


class PackedDatasetComponentsInstantiationModel(BaseModel):
    class PackedDatasetSettings(BaseModel):
        src_path: FilePath
        dst_path: Optional[Path] = None
        index_path: Optional[FilePath] = None
        jq_pattern: str
        num_cpus: Annotated[int, Field(strict=True, ge=1)] = os.cpu_count()
        eod_token: str
        processing_batch_size: Annotated[int, Field(strict=True, ge=1)]
        raw_samples_queue_size: Annotated[int, Field(strict=True, ge=1)]
        processed_samples_queue_size: Annotated[int, Field(strict=True, ge=1)]

    tokenizer: PydanticTokenizerIFType
    settings: PackedDatasetSettings


class TextGenerationInstantiationModel(BaseModel):
    class TextGenerationSettings(BaseModel):
        model_path: FilePath
        context_length: int
        device: PydanticPytorchDeviceType
        referencing_keys: Dict[str, str]

        @field_validator("device", mode="before")
        def parse_device(cls, device) -> PydanticPytorchDeviceType:
            return parse_torch_device(device)

    text_inference_component: PydanticTextInferenceComponentType
    settings: TextGenerationSettings