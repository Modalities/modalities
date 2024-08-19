import os
from pathlib import Path

import pytest

from modalities.config.config import load_app_config_dict
from modalities.models.model import NNModel
from modalities.models.utils import get_model_from_config, ModelTypeEnum
from tests.conftest import _ROOT_DIR


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture()
def config_file_path() -> Path:
    config_file_path = _ROOT_DIR / Path(
        "tests/fine_tuning/test_configs/" + "config_lorem_impsum_lora_loading_from_hf.yaml"
    )
    return config_file_path


@pytest.fixture()
def config_dict(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture()
def hf_model(set_env, config_dict: dict) -> NNModel:
    return get_model_from_config(config=config_dict, model_type=ModelTypeEnum.HUGGINGFACE_SMOL_LLM_MODEL)


@pytest.fixture()
def lora_model(set_env, config_dict: dict) -> NNModel:
    return get_model_from_config(config=config_dict, model_type=ModelTypeEnum.LORA_MODEL)


def test_loading_of_hf_model_from_config(lora_model):
    breakpoint()
