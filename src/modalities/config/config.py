import json
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field, FilePath, PositiveFloat, PositiveInt, confloat, conint, model_validator
from transformers import PretrainedConfig

from modalities.config.lookup_types import (
    BatchSamplerTypes,
    CheckpointingExectionTypes,
    CheckpointingStrategyTypes,
    CollatorTypes,
    DataloaderTypes,
    DatasetTypes,
    LossTypes,
    ModelTypes,
    OptimizerTypes,
    SamplerTypes,
    SchedulerTypes,
    TokenizerTypes,
)
from modalities.config.types import ProcessGroupBackendType
from modalities.models.gpt2.gpt2_model import GPT2Config
from modalities.models.huggingface.huggingface_models import HuggingFacePretrainedModelConfig
from modalities.running_env.fsdp.fsdp_running_env import RunningEnvConfig


class WandbConfig(BaseModel):
    class WandbMode(Enum):
        ONLINE = "ONLINE"
        OFFLINE = "OFFLINE"
        DISABLED = "DISABLED"

    project_name: str
    mode: WandbMode
    dir: Optional[Path] = Field(default_factory=lambda: Path("."))


class CudaKwargsConfig(BaseModel):
    num_workers: conint(ge=0)
    pin_memory: bool
    shuffle: bool


class TokenizerConfig(BaseModel):
    class GPT2TokenizerFastConfig(BaseModel):
        tokenizer_file: str  # FilePath not possible, since transformers.PretrainedTokenizers can only handle strings

    type_hint: TokenizerTypes
    config: GPT2TokenizerFastConfig


class DatasetConfig(BaseModel):
    class DummyDatasetConfig(BaseModel):
        num_samples: int
        sample_definition: List[Tuple[str, Tuple, str]]

    class MemMapDatasetConfig(BaseModel):
        raw_data_path: FilePath
        index_path: Optional[FilePath] = None
        block_size: conint(gt=0)
        tokenizer: TokenizerConfig
        jq_pattern: str
        sample_key: str

    class PackedMemMapDatasetContinuousConfig(BaseModel):
        raw_data_path: Path
        block_size: conint(gt=0)
        sample_key: str

    class PackedMemMapDatasetMegatronConfig(BaseModel):
        raw_data_path: Path
        block_size: conint(gt=0)
        sample_key: str

    class MMapIndexedDatasetConfig(BaseModel):
        path: Path
        skip_warmup: bool

    class OpenGPTXMMapDatasetConfig(BaseModel):
        num_samples: conint(ge=1)
        path: FilePath
        sample_key: str
        sequence_len: PositiveInt

    type_hint: DatasetTypes
    config: Union[
        DummyDatasetConfig,
        MemMapDatasetConfig,
        OpenGPTXMMapDatasetConfig,
        PackedMemMapDatasetContinuousConfig,
        PackedMemMapDatasetMegatronConfig,
        MMapIndexedDatasetConfig,
    ] = Field(union_mode="left_to_right")


class SamplerConfig(BaseModel):
    class DistributedSamplerConfig(BaseModel):
        rank: conint(ge=0)
        num_replicas: conint(ge=0)
        shuffle: bool

    type_hint: SamplerTypes
    config: DistributedSamplerConfig


class BatchSamplerConfig(BaseModel):
    class StandardBatchSamplerConfig(BaseModel):
        sampler: SamplerConfig
        batch_size: conint(gt=0)
        drop_last: bool

    type_hint: BatchSamplerTypes
    config: StandardBatchSamplerConfig


class CollatorConfig(BaseModel):
    class GPT2LLMCollatorConfig(BaseModel):
        sample_key: str
        target_key: str

    type_hint: CollatorTypes
    config: GPT2LLMCollatorConfig


class DataLoaderConfig(BaseModel):
    class LLMDataLoaderConfig(CudaKwargsConfig):
        dataloader_tag: str
        dataset: DatasetConfig
        batch_sampler: BatchSamplerConfig
        collate_fn: CollatorConfig

    type_hint: DataloaderTypes
    config: LLMDataLoaderConfig


class DataConfig(BaseModel):
    sample_key: str
    target_key: str
    sequence_len: int
    train_dataloader: DataLoaderConfig
    eval_dataloaders: List[DataLoaderConfig]


class ModelConfig(BaseModel):
    type_hint: ModelTypes
    config: HuggingFacePretrainedModelConfig | GPT2Config


class CLMCrossEntropyLossConfig(BaseModel):
    target_key: str
    prediction_key: str


class LossConfig(BaseModel):
    type_hint: LossTypes
    config: CLMCrossEntropyLossConfig


class TrainingConfig(BaseModel):
    # TODO: use this in Progress Logging
    global_num_training_samples: conint(gt=0)
    callback_interval_in_samples: conint(gt=0)
    process_group_backend: ProcessGroupBackendType
    local_rank: conint(ge=0)
    global_rank: conint(ge=0)
    world_size: conint(ge=0)
    main_rank: conint(ge=0)
    local_train_micro_batch_size: conint(gt=0)
    global_num_seen_samples: conint(ge=0)
    do_apply_activation_checkpointing: bool
    gradient_acc_step: conint(gt=0)

    @property
    def local_train_batch_size(self):
        return self.local_train_micro_batch_size * self.gradient_acc_step

    @property
    def global_train_batch_size(self):
        return self.local_train_batch_size * self.world_size

    @property
    def local_num_train_samples(self):
        exact = self.global_num_training_samples / self.world_size
        ret = self.global_num_training_samples // self.world_size
        if exact != ret:
            print(f"Calculated local_num_training_samples is not an integer. Clipping {exact} to {ret} ")
        return ret

    @property
    def local_num_seen_train_samples(self):
        exact = self.global_num_seen_samples / self.world_size
        ret = self.global_num_seen_samples // self.world_size
        if exact != ret:
            print(f"Calculated global_num_seen_samples is not an integer. Clipping {exact} to {ret} ")
        return ret

    @property
    def skip_num_local_train_batches(self) -> int:
        exact = self.global_num_seen_samples / self.world_size / self.local_train_micro_batch_size
        ret = self.global_num_seen_samples // self.world_size // self.local_train_micro_batch_size
        if exact != ret:
            print(f"Calculated skip_num_local_train_batches is not an integer. Clipping {exact} to {ret} ")
        return ret

    @property
    def num_training_batches(self) -> int:
        exact = self.global_num_training_samples / self.local_train_micro_batch_size
        ret = self.global_num_training_samples // self.local_train_micro_batch_size
        if exact != ret:
            warnings.warn(f"Calculated num_training_batches is not an integer. Clipping {exact} to {ret} ")
        return ret

    @property
    def callback_interval_in_batches_per_rank(self):
        exact = self.callback_interval_in_samples / self.local_train_micro_batch_size / self.world_size
        ret = max(self.callback_interval_in_samples // self.local_train_micro_batch_size // self.world_size, 1)
        if exact != ret:
            warnings.warn(
                f"Calculated callback_interval_in_batches_per_rank is not an integer. Clipping {exact} to {ret} "
            )
        return ret


class AdamWConfig(BaseModel):
    lr: confloat(ge=0.0)


class OptimizerConfig(BaseModel):
    type_hint: OptimizerTypes
    config: AdamWConfig


class OneCycleLRConfig(BaseModel):
    max_lr: PositiveFloat
    total_steps: conint(ge=1)
    pct_start: confloat(ge=0.0)
    anneal_strategy: str
    cycle_momentum: bool
    base_momentum: float | List
    max_momentum: float | List
    div_factor: PositiveFloat
    final_div_factor: PositiveFloat
    three_phase: bool
    last_epochs: int
    verbose: bool


class StepLRConfig(BaseModel):
    step_size: conint(ge=1)
    gamma: confloat(ge=0.0)


class ConstantLRConfig(BaseModel):
    factor: PositiveFloat
    total_iters: PositiveInt


class SchedulerConfig(BaseModel):
    type_hint: SchedulerTypes
    config: StepLRConfig | ConstantLRConfig | OneCycleLRConfig


class CheckpointingConfig(BaseModel):
    class CheckpointingStrategyConfig(BaseModel):
        class SaveEveryKStepsCheckpointingStrategyConfig(BaseModel):
            k: PositiveInt

        class SaveKMostRecentCheckpointsStrategyConfig(BaseModel):
            k: conint(ge=-1)

        type_hint: CheckpointingStrategyTypes
        config: SaveEveryKStepsCheckpointingStrategyConfig | SaveKMostRecentCheckpointsStrategyConfig

    class CheckpointingExecutionConfig(BaseModel):
        class FSDPToDiscCheckpointingConfig(BaseModel):
            checkpoint_path: Path
            global_rank: conint(ge=0)

        type_hint: CheckpointingExectionTypes
        config: FSDPToDiscCheckpointingConfig

    checkpointing_strategy: CheckpointingStrategyConfig
    checkpointing_execution: CheckpointingExecutionConfig


class RunMode(Enum):
    FROM_SCRATCH = "FROM_SCRATCH"
    WARM_START = "WARM_START"


class ModalitiesSetupConfig(BaseModel):
    class WarmStartSettings(BaseModel):
        checkpoint_model_path: Path
        global_num_seen_samples: conint(gt=0)
        checkpoint_optimizer_path: Optional[Path] = None
        checkpoint_lr_scheduler_path: Optional[Path] = None

    class FromScratchSettings(BaseModel):
        global_num_seen_samples: int = 0

    run_mode: RunMode
    settings: FromScratchSettings
    # settings: WarmStartSettings

    @model_validator(mode="after")
    def check_global_num_samples_equal_0_when_from_scratch(self) -> "ModalitiesSetupConfig":
        if self.run_mode == RunMode.FROM_SCRATCH:
            if self.settings.global_num_seen_samples != 0:
                raise ValueError("When starting from scratch, global_num_seen_samples must be 0.")
        return self


class AppConfig(BaseModel):
    modalities_setup: ModalitiesSetupConfig
    data: DataConfig
    training: TrainingConfig
    running_env: RunningEnvConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    checkpointing: CheckpointingConfig
    wandb: WandbConfig
    loss: LossConfig


class PretrainedGPTConfig(PretrainedConfig):
    model_type = "modalities_gpt2"

    def __init__(self, config: GPT2Config = None, **kwargs):
        if type(config) == dict:
            config = GPT2Config(**config)
        self.config = config

        super().__init__(**kwargs)

    def to_json_string(self, use_diff: bool = True) -> str:
        if self.config:
            json_dict = {"config": self.config.__dict__.copy(), "model_type": self.model_type}
            json_dict["config"]["attention"] = {
                "attention_type": self.config.attention.attention_type.value,
                "scaling_factor": self.config.attention.scaling_factor,
            }
            json_dict["config"]["weight_init"] = {
                "mean": self.config.weight_init.mean,
                "std": self.config.weight_init.std,
            }
        else:
            json_dict = {}
        return json.dumps(json_dict)
