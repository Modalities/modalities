import os
from pathlib import Path

import pytest
import torch
from pydantic import BaseModel

from modalities.config.config import load_app_config_dict
from modalities.models.model import NNModel
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from tests.conftest import _ROOT_DIR
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer

from modalities.__main__ import load_app_config_dict
from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import FSDPCheckpointLoading
from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import CheckpointingEntityType, FSDPCheckpointSaving
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2LLMConfig
from modalities.models.model_factory import ModelFactory
from modalities.optimizers.optimizer_factory import OptimizerFactory
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.training.training_progress import TrainingProgress


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture()
def initial_model_config_file_path():
    config_file_path = _ROOT_DIR / Path("tests/models/lora/test_configs/" + "lora_loading_from_initial_model.yaml")
    return config_file_path


@pytest.fixture()
def checkpointed_model_config_file_path() -> Path:
    config_file_path = _ROOT_DIR / Path("tests/models/lora/test_configs/" + "lora_loading_from_checkpointed_model.yaml")
    return config_file_path


@pytest.fixture()
def initial_model_config_dict(initial_model_config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=initial_model_config_file_path)


@pytest.fixture()
def checkpointed_model_config_dict_without_checkpoint_path(checkpointed_model_config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=checkpointed_model_config_file_path)


@pytest.fixture()
def initialized_model(set_env, checkpointed_model_config_dict_without_checkpoint_path: dict) -> NNModel:
    return get_model_from_config(
        config=checkpointed_model_config_dict_without_checkpoint_path, model_type=ModelTypeEnum.MODEL
    )


@pytest.fixture()
def checkpointed_model_config_dict_with_checkpoint_path(
    tmp_path: Path,
    initialized_model: NNModel,
    checkpointed_model_config_dict_without_checkpoint_path: dict,
) -> dict:
    model_file_path = tmp_path / "pytorch_model.bin"
    torch.save(initialized_model.state_dict(), model_file_path)
    # Adding the checkpoint path in tmp folder to the config dict
    config_dict_with_checkpoint_path = checkpointed_model_config_dict_without_checkpoint_path
    config_dict_with_checkpoint_path["checkpointed_model"]["config"]["checkpoint_path"] = model_file_path
    return config_dict_with_checkpoint_path


@pytest.fixture()
def checkpointed_lora_model(set_env, checkpointed_model_config_dict_with_checkpoint_path: dict) -> NNModel:
    return get_model_from_config(
        config=checkpointed_model_config_dict_with_checkpoint_path, model_type=ModelTypeEnum.LORA_MODEL
    )


@pytest.fixture()
def initial_lora_model(set_env, initial_model_config_dict: dict) -> NNModel:
    return get_model_from_config(config=initial_model_config_dict, model_type=ModelTypeEnum.LORA_MODEL)


def test_load_lora_model_from_checkpointed_model(
    checkpointed_lora_model: NNModel, checkpointed_model_config_dict_without_checkpoint_path: dict
):
    target_layer_class_names = checkpointed_model_config_dict_without_checkpoint_path["lora_model"]["config"][
        "target_layers"
    ]
    for module in list(checkpointed_lora_model.modules()):
        assert type(module).__name__ not in target_layer_class_names


def test_load_lora_model_from_initial_model(initial_lora_model: NNModel, initial_model_config_dict: dict):
    target_layer_class_names = initial_model_config_dict["lora_model"]["config"]["target_layers"]
    for module in list(initial_lora_model.modules()):
        assert type(module).__name__ not in target_layer_class_names


@pytest.fixture()
def checkpoint_path():
    return "data/checkpoints/lora_training.yaml"


@pytest.fixture()
def checkpoint_config_dict(checkpoint_path):
    return load_app_config_dict(config_file_path=checkpoint_path)


def lora_model_from_checkpointed_lora_model(checkpoint_config_dict):
    # todo double check if this works
    class LoraInstantiatedModel(BaseModel):
        model: PydanticPytorchModuleType

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    components = component_factory.build_components(
        config_dict=checkpoint_config_dict, components_model_type=LoraInstantiatedModel
    )
    model = components.model
    return model


def test_load_lora_model_from_checkpointed_lora_model(checkpoint_path):
    # todo
    # 1. load model from data/checkpoints/lora_training.yaml and the corresponding bin files
    # 2. check if the model is a lora model
    ...
