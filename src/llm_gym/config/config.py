import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, DirectoryPath, PositiveFloat, PositiveInt, confloat, conint, model_validator, FilePath

from llm_gym.config.lookup_types import (
    DataLoaderTypes,
    LossTypes,
    ModelTypes,
    OptimizerTypes,
    SamplerTypes,
    SchedulerTypes,
)
from llm_gym.config.types import ProcessGroupBackendType
from llm_gym.fsdp.fsdp_running_env import RunningEnv, RunningEnvConfig
from llm_gym.models.gpt2.gpt2_model import GPTConfig


class WandbConfig(BaseModel):
    project_name: str


class DistributedSamplerConfig(BaseModel):
    rank: conint(ge=0)
    num_replicas: conint(ge=0)
    shuffle: bool


class SamplerConfig(BaseModel):
    type_hint: SamplerTypes
    config: DistributedSamplerConfig


class DatasetConfig(BaseModel):
    num_samples: conint(ge=1)
    path: FilePath
    sample_key: str
    sequence_len: PositiveInt


class LLMDataLoaderConfig(BaseModel):
    dataset_tag: str
    batch_size: conint(ge=1)
    num_workers: conint(ge=1)
    pin_memory: bool
    shuffle: bool
    sampler: SamplerConfig
    dataset: DatasetConfig


class DataLoaderConfig(BaseModel):
    type_hint: DataLoaderTypes
    config: LLMDataLoaderConfig


class DataConfig(BaseModel):
    dataset_dir_path: DirectoryPath
    sample_key: str
    target_key: str
    sequence_len: PositiveInt
    train_dataloader: DataLoaderConfig
    eval_dataloaders: List[DataLoaderConfig]


class TrainingConfig(BaseModel):
    process_group_backend: ProcessGroupBackendType
    local_rank: conint(ge=0)
    global_rank: conint(ge=0)
    world_size: conint(ge=0)
    main_rank: conint(ge=0)
    eval_interval_in_batches: conint(ge=1)
    num_training_samples: int

    @property
    def eval_interval_per_rank(self):
        return self.eval_interval_in_batches // self.world_size

    @property
    def num_training_batches_per_rank(self):
        return self.num_training_batches // self.world_size

    # @model_validator(mode="after")
    # def validate_multiples(self) -> "TrainingConfig":
    #     computed_num_training_batches = self.eval_interval_per_rank * self.world_size * self.eval_interval_in_batches
    #     if computed_num_training_batches != self.num_training_batches:
    #         raise ValueError(
    #             f"eval_interval_per_rank * world_size * eval_interval_in_batches ({computed_num_training_batches})"
    #             f" != num_training_batches ({self.num_training_batches})"
    #         )
    #     return self


class GPT2Config(BaseModel):
    config: GPTConfig


class ModelConfig(BaseModel):
    type_hint: ModelTypes
    config: GPT2Config


class CLMCrossEntropyLossConfig(BaseModel):
    target_key: str
    prediction_key: str


class LossConfig(BaseModel):
    type_hint: LossTypes
    config: CLMCrossEntropyLossConfig


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


class CheckpointConfig(BaseModel):
    checkpointing_rank: conint(ge=0, le=os.environ.get("WORLD_SIZE", 0))
    dir_path: Path


class AppConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    running_env: RunningEnvConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    checkpoint: CheckpointConfig
    wandb: WandbConfig
