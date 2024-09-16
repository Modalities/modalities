import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator, root_validator

from modalities.config.pydanctic_if_types import (
    PydanticCheckpointSavingIFType,
    PydanticDatasetIFType,
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
from modalities.dataloader.dataset import Dataset
from modalities.util import warn_rank_0


class CudaEnvSettings(BaseModel):
    local_rank: Annotated[int, Field(strict=True, ge=0)]
    world_size: Annotated[int, Field(strict=True, ge=1)]
    global_rank: Annotated[int, Field(strict=True, ge=0)]


class StepProfile(BaseModel):
    gradient_accumulation_steps: Annotated[int, Field(strict=True, ge=1)]
    local_train_micro_batch_size: Annotated[int, Field(strict=True, ge=1)]
    sequence_length: Annotated[int, Field(strict=True, ge=1)]


class ConsistencyEnforcement(BaseModel):
    enforce_tokens_per_step_consistency: bool = True
    enforce_last_step_logged: bool = True
    enforce_last_step_evaluated: bool = True
    enforce_last_step_checkpointed: bool = True


class Intervals(BaseModel):
    training_log_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
    checkpointing_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
    evaluation_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]


class TrainingTarget(BaseModel):
    num_target_tokens: Annotated[int, Field(strict=True, ge=1)]
    num_target_steps: Annotated[int, Field(strict=True, ge=1)]


class TrainingProgress(BaseModel):
    global_num_seen_tokens: Annotated[int, Field(strict=True, ge=0)]
    num_seen_steps: Annotated[int, Field(strict=True, ge=0)]
    local_num_seen_batches: Annotated[int, Field(strict=True, ge=0)]
    last_step: Annotated[int, Field(strict=True, ge=-1)]


class TrainingComponentsInstantiationModel(BaseModel):
    class Settings(BaseModel):
        class Paths(BaseModel):
            checkpoint_saving_path: Path  # Explicitly defined field

            class Config:
                extra = "allow"

            @root_validator(pre=True)
            def _validate_all_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
                for field_name, value in values.items():
                    if isinstance(value, str):  # If a value is a string, convert it to Path
                        values[field_name] = Path(value)
                    elif not isinstance(value, Path):
                        raise TypeError(f"Field '{field_name}' must be of type Path, but got {type(value)} instead.")
                return values

        class WarmstartCheckpointPaths(BaseModel):
            model_checkpoint_path: Path
            optimizer_checkpoint_path: Path

        experiment_id: str
        config_file_path: FilePath
        referencing_keys: Dict[str, str]
        cuda_env: CudaEnvSettings
        paths: Paths
        intervals: Intervals
        consistency_enforcement: ConsistencyEnforcement
        step_profile: StepProfile
        training_target: TrainingTarget
        training_progress: TrainingProgress
        warmstart_checkpoint_paths: Optional[WarmstartCheckpointPaths] = None

        @model_validator(mode="after")
        def _check_tokens_per_step_conistency(self) -> "TrainingComponentsInstantiationModel.Settings":
            # Check if the number of target steps and target tokens are consistent with the step profile
            required_num_tokens_per_step = (
                self.training_target.num_target_tokens - self.training_progress.global_num_seen_tokens
            ) / (self.training_target.num_target_steps - self.training_progress.num_seen_steps)
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
                if self.consistency_enforcement.enforce_tokens_per_step_consistency:
                    raise ValueError(warning_message)
                warn_rank_0(warning_message)
            return self

        @model_validator(mode="after")
        def _check_last_step_logged(self) -> "TrainingComponentsInstantiationModel.Settings":
            # Check if the training is logged after the last step
            remaining_steps = self.training_target.num_target_steps - self.training_progress.num_seen_steps
            if remaining_steps % self.intervals.training_log_interval_in_steps != 0:
                warning_message = (
                    "Last step will not be logged. Since remaining_steps "
                    f"({remaining_steps}) "
                    "is not a multiple of training_log_interval_in_steps "
                    f"({self.intervals.training_log_interval_in_steps})"
                )
                if self.consistency_enforcement.enforce_last_step_logged:
                    raise ValueError(warning_message)
                warn_rank_0(warning_message)
            return self

        @model_validator(mode="after")
        def _check_last_step_evaluated(self) -> "TrainingComponentsInstantiationModel.Settings":
            # Check if the model is evaluated after the last step
            remaining_steps = self.training_target.num_target_steps - self.training_progress.num_seen_steps
            if remaining_steps % self.intervals.evaluation_interval_in_steps != 0:
                warning_message = (
                    "Last step will not be evaluated. Since remaining_steps "
                    f"({remaining_steps}) "
                    "is not a multiple of evaluation_interval_in_steps "
                    f"({self.intervals.evaluation_interval_in_steps})"
                )
                if self.consistency_enforcement.enforce_last_step_evaluated:
                    raise ValueError(warning_message)
                warn_rank_0(warning_message)
            return self

        @model_validator(mode="after")
        def _check_last_step_checkpointed(self) -> "TrainingComponentsInstantiationModel.Settings":
            # Check if the model is evaluated after the last step
            remaining_steps = self.training_target.num_target_steps - self.training_progress.num_seen_steps
            if remaining_steps % self.intervals.checkpointing_interval_in_steps != 0:
                warning_message = (
                    "Last step will not be checkpointed. Since remaining_steps "
                    f"({remaining_steps}) "
                    "is not a multiple of checkpointing_interval_in_steps "
                    f"({self.intervals.checkpointing_interval_in_steps})"
                )
                if self.consistency_enforcement.enforce_last_step_checkpointed:
                    raise ValueError(warning_message)
                warn_rank_0(warning_message)
            return self

    settings: Settings
    wrapped_model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType
    scheduler: PydanticLRSchedulerIFType
    loss_fn: PydanticLossIFType
    train_dataset: PydanticDatasetIFType
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: List[PydanticLLMDataLoaderIFType]
    progress_subscriber: PydanticMessageSubscriberIFType
    evaluation_subscriber: PydanticMessageSubscriberIFType
    checkpoint_saving: PydanticCheckpointSavingIFType
    gradient_clipper: PydanticGradientClipperIFType

    @model_validator(mode="after")
    def _check_token_amount_in_dataset(self) -> "TrainingComponentsInstantiationModel.Settings":
        if (
            len(self.train_dataset) * self.settings.step_profile.sequence_length
            < self.settings.training_target.num_target_tokens
        ):
            raise ValueError(
                "Not enough tokens in the dataset. "
                f"Actual: {len(self.train_dataset) * self.settings.step_profile.sequence_length}, "
                f"Expected: >={self.settings.training_target.num_target_tokens}"
            )
        return self


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


class TrainingReportGenerator:
    def __init__(
        self,
        training_target: TrainingTarget,
        intervals: Intervals,
        step_profile: StepProfile,
        cuda_env: CudaEnvSettings,
        consistency_enforcement: ConsistencyEnforcement,
        train_dataset: Dataset,
        training_progress: TrainingProgress,
    ):
        self.training_target = training_target
        self.intervals = intervals
        self.step_profile = step_profile
        self.cuda_env = cuda_env
        self.train_dataset = train_dataset
        self.consistency_enforcement = consistency_enforcement
        self.training_progress = training_progress

    def get_report(self) -> str:
        def _get_formatted_dict_str(d: Dict[str, Any]) -> str:
            return "\n\t".join([f"{k}: {v}" for k, v in d.items()])

        def _get_formatted_list_str(lst: List[str]) -> str:
            return "\n\t".join(lst)

        training_target_str = _get_formatted_dict_str(dict(self.training_target))
        interval_str = _get_formatted_dict_str(dict(self.intervals))
        step_profile_str = _get_formatted_dict_str(dict(self.step_profile))
        cuda_env_str = _get_formatted_dict_str(dict(self.cuda_env))
        consistency_enforcement_str = _get_formatted_dict_str(dict(self.consistency_enforcement))
        training_progress_str = _get_formatted_dict_str(dict(self.training_progress))
        warnings_str = _get_formatted_list_str(self._get_issue_warnings())

        report = (
            "\n\n\n======================== Training Report ========================\n"
            f"Training target: \n\t{training_target_str} \n"
            f"Intervals: \n\t{interval_str}\n"
            f"Step profile: \n\t{step_profile_str}\n"
            f"CUDA environment settings: \n\t{cuda_env_str}\n"
            f"Consistency enforcement: \n\t{consistency_enforcement_str}\n"
            f"Training progress: \n\t{training_progress_str}\n"
            f"Warnings: \n\t\033[38;5;214m{warnings_str} \033[0m \n"
            "====================================================================\n\n\n"
        )
        return report

    def _get_issue_warnings(self) -> List[str]:
        issue_warnings = []
        num_tokens = (
            self.step_profile.local_train_micro_batch_size
            * self.step_profile.sequence_length
            * self.step_profile.gradient_accumulation_steps
            * self.cuda_env.world_size
            * self.training_target.num_target_steps
        )
        # Check if the number of target tokens and (number of tokens per step * num steps) are consistent
        if self.training_target.num_target_tokens != num_tokens:
            missing_percentage = (1 - num_tokens / self.training_target.num_target_tokens) * 100
            issue_warnings.append(
                f"Number of target tokens ({self.training_target.num_target_tokens}) "
                f"does not match the number of tokens per step * num steps ({num_tokens})."
                f"Missing {missing_percentage:.2f}% of target tokens."
            )

        # Check if the number of tokens in the dataset and the number of target tokens are consistent
        tokens_in_dataset = len(self.train_dataset) * self.step_profile.sequence_length
        if tokens_in_dataset != self.training_target.num_target_tokens:
            missing_percentage = (1 - num_tokens / tokens_in_dataset) * 100
            issue_warnings.append(
                f"Number of tokens in the dataset ({tokens_in_dataset}) "
                f"does not match the number of target tokens ({self.training_target.num_target_tokens}). "
                f"Missing {missing_percentage:.2f}% of tokens in the dataset."
            )

        # Check if the training is logged after the last step
        remaining_steps = self.training_target.num_target_steps - self.training_progress.num_seen_steps
        if remaining_steps % self.intervals.training_log_interval_in_steps != 0:
            issue_warnings.append(
                f"Last step will not be logged. Since remaining_steps "
                f"({remaining_steps}) "
                "is not a multiple of training_log_interval_in_steps "
                f"({self.intervals.training_log_interval_in_steps})."
            )

        # Check if the model is evaluated after the last step
        if remaining_steps % self.intervals.evaluation_interval_in_steps != 0:
            issue_warnings.append(
                f"Last step will not be evaluated. Since remaining_steps "
                f"({remaining_steps}) "
                "is not a multiple of evaluation_interval_in_steps "
                f"({self.intervals.evaluation_interval_in_steps})."
            )
        # Check if the model is checkpointed after the last step
        if remaining_steps % self.intervals.checkpointing_interval_in_steps != 0:
            issue_warnings.append(
                f"Last step will not be checkpointed. Since remaining_steps "
                f"({remaining_steps}) "
                "is not a multiple of checkpointing_interval_in_steps "
                f"({self.intervals.checkpointing_interval_in_steps})."
            )

        return issue_warnings
