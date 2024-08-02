import os
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator

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
            training_log_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            checkpointing_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            evaluation_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            activation_checkpointing_modules: Optional[List[str]] = Field(default_factory=list)
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
    loss_fn: PydanticLossIFType
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
        sequence_length: int
        device: PydanticPytorchDeviceType
        referencing_keys: Dict[str, str]

        # avoid warning about protected namespace 'model_', see
        # https://docs.pydantic.dev/2.7/api/config/#pydantic.config.ConfigDict.protected_namespaces
        model_config = ConfigDict(protected_namespaces=())

        @field_validator("device", mode="before")
        def parse_device(cls, device) -> PydanticPytorchDeviceType:
            return parse_torch_device(device)

    text_inference_component: PydanticTextInferenceComponentType
    settings: TextGenerationSettings
