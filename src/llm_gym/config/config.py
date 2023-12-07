import os
from pathlib import Path
from typing import Dict, List, Text

from pydantic import BaseModel, DirectoryPath, FilePath, PositiveFloat, PositiveInt, confloat, conint

from llm_gym.config.lookup_types import (
    CollatorTypes,
    DataloaderTypes,
    DatasetTypes,
    LossTypes,
    ModelTypes,
    OptimizerTypes,
    SamplerTypes,
    SchedulerTypes,
)
from llm_gym.config.types import ProcessGroupBackendType
from llm_gym.fsdp.fsdp_runner import RunnerConfig
from llm_gym.models.gpt2.gpt2_model import GPTConfig


class WandbConfig(BaseModel):
    project_name: str


class CudaKwargsConfig(BaseModel):
    num_workers: conint(ge=1)
    pin_memory: bool
    shuffle: bool


class DatasetConfig(BaseModel):
    # TODO: extend this with packed MemMapDataset / MegatronLMs-based packed version
    class MemMapDatasetConfig(BaseModel):
        raw_data_path: DirectoryPath | FilePath
        block_size: conint(gt=0)
        tokenizer_path: FilePath
        jq_pattern: str

    class PackedMemMapDatasetContinuousConfig(BaseModel):
        raw_data_path: DirectoryPath | FilePath
        block_size: conint(gt=0)

    class PackedMemMapDatasetMegatronConfig(BaseModel):
        raw_data_path: DirectoryPath | FilePath
        block_size: conint(gt=0)

    type_hint: DatasetTypes
    config: MemMapDatasetConfig | PackedMemMapDatasetContinuousConfig | PackedMemMapDatasetMegatronConfig


class SamplerConfig(BaseModel):
    class DistributedSamplerConfig(BaseModel):
        rank: conint(ge=0)
        num_replicas: conint(ge=0)
        shuffle: bool

    type_hint: SamplerTypes
    config: DistributedSamplerConfig


class CollatorConfig(BaseModel):
    class GPT2LLMCollatorConfig(BaseModel):
        sample_key: str
        target_key: str

    type_hint: CollatorTypes
    config: GPT2LLMCollatorConfig


class DataLoaderConfig(BaseModel):
    class LLMDataLoaderConfig(CudaKwargsConfig):
        batch_size: conint(gt=0)
        dataloader_tag: str
        dataset: DatasetConfig
        sampler: SamplerConfig
        collate_fn: CollatorConfig

    type_hint: DataloaderTypes
    config: LLMDataLoaderConfig


class TrainingConfig(BaseModel):
    train_dataloader: DataLoaderConfig
    evaluation_dataloaders: Dict[Text, DataLoaderConfig]
    # TODO: use this in Progress Logging
    num_training_samples: conint(gt=0)
    callback_interval_in_samples: conint(gt=0)
    process_group_backend: ProcessGroupBackendType
    local_rank: conint(ge=0)
    global_rank: conint(ge=0)
    world_size: conint(ge=0)
    main_rank: conint(ge=0)

    @property
    def num_training_batches(self) -> int:
        exact = self.num_training_samples / self.train_dataloader.config.batch_size
        ret = self.num_training_samples // self.train_dataloader.config.batch_size
        if exact != ret:
            print(f"Calculated num_training_batches is not an integer. Clipping {exact} to {ret} ")
        return ret

    @property
    def eval_interval_in_batches_per_rank(self):
        exact = self.callback_interval_in_samples / self.train_dataloader.config.batch_size / self.world_size
        ret = self.callback_interval_in_samples // self.train_dataloader.config.batch_size // self.world_size
        if exact != ret:
            print(f"Calculated eval_interval_in_batches_per_rank is not an integer. Clipping {exact} to {ret} ")
        return ret

    @property
    def num_batches_per_rank(self):
        exact = self.num_training_batches / self.world_size
        ret = self.num_training_batches // self.world_size
        if exact != ret:
            print(f"Calculated eval_interval_in_batches is not an integer. Clipping {exact} to {ret} ")
        return ret


# TODO: remove this?? Seems unnecessary to add another composition layer here
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
    training: TrainingConfig
    loss: LossConfig
    runner: RunnerConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    checkpoint: CheckpointConfig
    wandb: WandbConfig
