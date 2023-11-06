import traceback
from enum import Enum
from typing import Annotated, List

from hydra._internal.utils import _locate
from pydantic import BaseModel, DirectoryPath, conint
from pydantic.functional_validators import AfterValidator


def validate_class_path(path: str):
    try:
        _locate(path)
    except Exception as hydra_error:
        raise ValueError(
            f"Could not resolve path to class {path}.",
        ) from hydra_error
    return path


TargetPath = Annotated[str, AfterValidator(validate_class_path)]


class ProcessGroupBackendEnum(str, Enum):
    nccl = "nccl"


class DataConfig(BaseModel):
    dataset_dir_path: DirectoryPath


class TrainingConfig(BaseModel):
    num_epochs: conint(gt=0)
    process_group_backend: ProcessGroupBackendEnum


class ModelConfig(BaseModel):
    target_class: TargetPath
    prediction_publication_key: str


class LossConfig(BaseModel):
    target_class: TargetPath
    target_subscription_key: str
    prediction_subscription_key: str


class RunnerConfig(BaseModel):
    target_class: TargetPath
    process_group_backend: ProcessGroupBackendEnum


class AppConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    runner: RunnerConfig
    model: ModelConfig
