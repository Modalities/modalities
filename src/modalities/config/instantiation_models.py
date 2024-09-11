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


class TrainingComponentsInstantiationModel(BaseModel):
    class Settings(BaseModel):
        class CudaEnvSettings(BaseModel):
            local_rank: Annotated[int, Field(strict=True, ge=0)]
            world_size: Annotated[int, Field(strict=True, ge=1)]
            global_rank: Annotated[int, Field(strict=True, ge=0)]

        class Intervals(BaseModel):
            training_log_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            checkpointing_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
            evaluation_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]

        class TrainingTarget(BaseModel):
            num_target_tokens: Annotated[int, Field(strict=True, ge=1)]
            num_target_steps: Annotated[int, Field(strict=True, ge=1)]

        class StepProfile(BaseModel):
            gradient_accumulation_steps: Annotated[int, Field(strict=True, ge=1)]
            local_train_micro_batch_size: Annotated[int, Field(strict=True, ge=1)]
            sequence_length: Annotated[int, Field(strict=True, ge=1)]

        class Paths(BaseModel):
            checkpoint_saving_path: Path

        class Warmstart(BaseModel):
            class TrainingProgress(BaseModel):
                global_num_seen_tokens: Annotated[int, Field(strict=True, ge=1)]
                num_seen_steps: Annotated[int, Field(strict=True, ge=1)]
                local_num_seen_batches: Annotated[int, Field(strict=True, ge=0)]
                last_step: Annotated[int, Field(strict=True, ge=0)]

            class CheckpointPaths(BaseModel):
                model_checkpoint_path: Path
                optimizer_checkpoint_path: Path

            training_progress: TrainingProgress
            checkpoint_paths: CheckpointPaths

        experiment_id: str
        config_file_path: FilePath
        referencing_keys: Dict[str, str]
        cuda_env: CudaEnvSettings
        paths: Paths
        intervals: Intervals
        enforce_tokens_per_step_conistency: bool = True
        step_profile: StepProfile
        training_target: TrainingTarget
        warmstart: Optional[Warmstart] = None

        @model_validator(mode="after")
        def _check_tokens_per_step_conistency(self) -> "TrainingComponentsInstantiationModel.Settings":
            # Check if the number of target steps and target tokens are consistent with the step profile
            required_num_tokens_per_step = (
                self.training_target.num_target_tokens / self.training_target.num_target_steps
            )
            step_profile_num_tokens_per_step = (
                self.step_profile.local_train_micro_batch_size
                * self.step_profile.sequence_length
                * self.step_profile.gradient_accumulation_steps
                * self.cuda_env.world_size
            )
            if required_num_tokens_per_step != step_profile_num_tokens_per_step:
                warning_message = (
                    f"Required number of tokens per step is ({required_num_tokens_per_step}) "
                    f"which does not match the number of tokens per step ({step_profile_num_tokens_per_step}) "
                    "from the step profile."
                )
                if self.enforce_tokens_per_step_conistency:
                    raise ValueError(warning_message)
                warnings.warn(warning_message)
            return self

    settings: Settings
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
