import os
from pathlib import Path

import pytest
import torch

from modalities.config.config import load_app_config_dict
from modalities.models.model import NNModel
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from tests.conftest import _ROOT_DIR


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture()
def config_file_path() -> Path:
    config_file_path = _ROOT_DIR / Path(
        "tests/fine_tuning/test_configs/" + "config_lorem_impsum_lora_loading.yaml"
    )
    return config_file_path


@pytest.fixture()
def config_dict_without_checkpoint_path(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture()
def initialized_model(set_env, config_dict_without_checkpoint_path: dict) -> NNModel:
    return get_model_from_config(
        config=config_dict_without_checkpoint_path, model_type=ModelTypeEnum.MODEL
    )


@pytest.fixture()
def config_dict_with_checkpoint_path(
        tmp_path: Path,
        initialized_model: NNModel,
        config_dict_without_checkpoint_path: dict,
) -> dict:
    model_file_path = tmp_path / "pytorch_model.bin"
    torch.save(initialized_model.state_dict(), model_file_path)

    # Adding the checkpoint path in tmp folder to the config dict
    config_dict_with_checkpoint_path = config_dict_without_checkpoint_path
    config_dict_with_checkpoint_path["checkpointed_model"]["config"][
        "checkpoint_path"
    ] = model_file_path
    return config_dict_with_checkpoint_path


@pytest.fixture()
def lora_model(set_env, config_dict_with_checkpoint_path: dict) -> NNModel:
    return get_model_from_config(
        config=config_dict_with_checkpoint_path, model_type=ModelTypeEnum.LORA_MODEL
    )


def test_load_lora_model(lora_model: NNModel, config_dict_without_checkpoint_path: dict):
    target_layer_class_names = config_dict_without_checkpoint_path["lora_model"][
        "config"
    ]["target_layers"]
    for module in list(lora_model.modules()):
        assert type(module).__name__ not in target_layer_class_names
