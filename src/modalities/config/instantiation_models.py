import os
import warnings
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator

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
        class Intervals:
            training_log_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            checkpointing_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            evaluation_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]

        class TrainingProfile(BaseModel):
            num_training_tokens: Annotated[int, Field(strict=True, ge=1)]
            num_target_steps: Annotated[int, Field(strict=True, ge=1)]

        class StepProfile(BaseModel):
            gradient_accumulation_steps: Annotated[int, Field(strict=True, ge=1)]
            local_train_micro_batch_size: Annotated[int, Field(strict=True, ge=1)]
            sequence_length: Annotated[int, Field(strict=True, ge=1)]

        class Training(BaseModel):
            training_profile: "TrainingComponentsInstantiationModel.TrainingSettings.TrainingProfile"
            step_profile: "TrainingComponentsInstantiationModel.TrainingSettings.StepProfile"

        class Paths(BaseModel):
            checkpoint_saving_path: Path

        class Warmstart(BaseModel):
            class TrainingProgress(BaseModel):
                global_num_seen_tokens: Annotated[int, Field(strict=True, ge=1)]
                num_seen_steps: Annotated[int, Field(strict=True, ge=1)]
                skip_num_batches: Annotated[int, Field(strict=True, ge=0)]
                last_step: Annotated[int, Field(strict=True, ge=0)]

            class CheckpointPaths(BaseModel):
                model_checkpoint_path: Path
                optimizer_checkpoint_path: Path

            enforce_tokens_per_step_conistency: bool = True
            step_profile: "TrainingComponentsInstantiationModel.TrainingSettings.StepProfile"
            training_profile: "TrainingComponentsInstantiationModel.TrainingSettings.TrainingProfile"
            training_progress: TrainingProgress
            checkpoint_paths: CheckpointPaths

        experiment_id: str
        referencing_keys: Dict[str, str]
        intervals: Intervals
        training: Optional[Training] = None
        warmstart: Optional[Warmstart] = None
        cuda_env: CudaEnvSettings
        paths: Paths

        @model_validator(mode="after")
        def _check_tokens_per_step_conistency(self) -> "TrainingComponentsInstantiationModel.TrainingSettings":
            # Check if the number of tokens per step are consistent in initial training run and warmstart
            if self.warmstart is not None:
                previous_num_tokens_per_step = self.training.num_training_tokens / self.training.num_target_steps
                current_num_tokens_per_step = (
                    self.training.local_train_micro_batch_size
                    * self.training.sequence_length
                    * self.training.gradient_acc_steps
                    * self.cuda_env.world_size
                )
                if previous_num_tokens_per_step != current_num_tokens_per_step:
                    warning_message = (
                        f"Number of tokens per step in previous run ({previous_num_tokens_per_step}) "
                        f"and current warmstart ({current_num_tokens_per_step}) do not match."
                    )
                    if self.warmstart.enforce_tokens_per_step_conistency:
                        raise ValueError(warning_message)
                    warnings.warn(warning_message)
            return self

        @model_validator(mode="after")
        def _check_training_or_warmstart_set(self) -> "TrainingComponentsInstantiationModel.TrainingSettings":
            # Check if either training or warmstart settings are provided
            if self.warmstart is None and self.training is None:
                raise ValueError("Either training or warmstart settings must be provided.")
            return self

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
