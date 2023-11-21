import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, DirectoryPath, PositiveFloat, PositiveInt, confloat, conint, model_validator

from llm_gym.config.lookup_types import LossTypes, ModelTypes, OptimizerTypes, SchedulerTypes
from llm_gym.config.types import ProcessGroupBackendType
from llm_gym.fsdp.fsdp_runner import RunnerConfig
from llm_gym.models.gpt2.gpt2_model import GPTConfig


class WandbConfig(BaseModel):
    project_name: str


class CudaKwargsConfig(BaseModel):
    num_workers: conint(ge=1)
    pin_memory: bool
    shuffle: bool


class DataLoaderConfig(BaseModel):
    train_dataset_tag: str
    val_dataset_tag: str
    test_dataset_tag: str
    cuda_kwargs: CudaKwargsConfig


class DataConfig(BaseModel):
    dataset_dir_path: DirectoryPath
    sample_key: str
    target_key: str
    sequence_len: PositiveInt
    dataloader: DataLoaderConfig


class TrainingConfig(BaseModel):
    num_training_batches: conint(gt=0)
    process_group_backend: ProcessGroupBackendType
    num_batches_per_training_sequence: int
    local_rank: conint(ge=0)
    global_rank: conint(ge=0)
    world_size: conint(ge=0)
    main_rank: conint(ge=0)
    eval_interval_in_batches: conint(ge=1)
    training_batch_size: int
    evaluation_batch_size: int
    test_batch_size: int

    @property
    def eval_interval_per_rank(self):
        return self.num_training_batches // self.eval_interval_in_batches // self.world_size

    @property
    def num_batches_per_rank(self):
        return self.num_training_batches // self.world_size

    @model_validator(mode="after")
    def validate_multiples(self) -> "TrainingConfig":
        computed_num_training_batches = self.eval_interval_per_rank * self.world_size * self.eval_interval_in_batches
        if computed_num_training_batches != self.num_training_batches:
            raise ValueError(
                "num_batches_per_training_sequence_per_rank * world_size * num_batches_per_training_sequence"
                " != num_training_batches"
            )
        return self


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
    pct_start: confloat(ge=0.)
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
    runner: RunnerConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    checkpoint: CheckpointConfig
    wandb: WandbConfig
