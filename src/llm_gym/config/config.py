from enum import Enum
from typing import Annotated

from hydra._internal.utils import _locate
from pydantic import BaseModel, DirectoryPath, FilePath, conint, model_validator
from pydantic.functional_validators import AfterValidator

from llm_gym.models.gpt2.gpt2_model import GPTConfig


def validate_class_path(path: str):
    try:
        _locate(path)
    except Exception as hydra_error:
        raise ValueError(
            f"Could not resolve path to class {path}.",
        ) from hydra_error
    return path


ClassPath = Annotated[str, AfterValidator(validate_class_path)]


class ProcessGroupBackendEnum(str, Enum):
    nccl = "nccl"


class DataConfig(BaseModel):
    dataset_dir_path: DirectoryPath | FilePath


class TrainingConfig(BaseModel):
    num_training_batches: conint(gt=0)
    process_group_backend: ProcessGroupBackendEnum


class ModelConfig(BaseModel):
    target_class: ClassPath
    prediction_publication_key: str
    config: GPTConfig


class LossConfig(BaseModel):
    target_class: ClassPath
    target_subscription_key: str
    prediction_subscription_key: str


class RunnerConfig(BaseModel):
    target_class: ClassPath
    process_group_backend: ProcessGroupBackendEnum


class GlobalsConfig(BaseModel):
    local_rank: int
    global_rank: int
    world_size: int
    num_training_batches: int
    num_batches_per_training_sequence: int
    training_batch_size: int
    evaluation_batch_size: int

    @property
    def num_batches_per_training_sequence_per_rank(self):
        return self.num_training_batches // self.num_batches_per_training_sequence // self.world_size

    @property
    def num_batches_per_rank(self):
        return self.num_training_batches // self.world_size

    @model_validator(mode="after")
    def validate_multiples(self) -> "GlobalsConfig":
        computed_num_training_batches = (
            self.num_batches_per_training_sequence_per_rank * self.world_size * self.num_batches_per_training_sequence
        )
        if computed_num_training_batches != self.num_training_batches:
            raise ValueError(
                "num_batches_per_training_sequence_per_rank * world_size * num_batches_per_training_sequence != num_training_batches"  # noqa: E501
            )
        return self


class AppConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    runner: RunnerConfig
    model: ModelConfig
    globals: GlobalsConfig
