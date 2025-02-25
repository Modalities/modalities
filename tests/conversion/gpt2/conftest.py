import os
import shutil
from pathlib import Path

import pytest
import torch

from modalities.config.config import load_app_config_dict
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from tests.conftest import _ROOT_DIR


@pytest.fixture()
def gpt2_config_path(tmp_path: Path, initialized_model: GPT2LLM, config_file_path: str) -> str:
    new_config_filename = tmp_path / "gpt2_config_test.yaml"
    model_path = tmp_path / "model.pth"
    shutil.copy(config_file_path, new_config_filename)
    torch.save(initialized_model.state_dict(), model_path)
    with open(new_config_filename, "r") as file:
        content = file.read()
    content = content.replace("checkpoint_path: null", f"checkpoint_path: {model_path}")
    with open(new_config_filename, "w") as file:
        file.write(content)
    return str(new_config_filename)


@pytest.fixture()
def initialized_model(set_env, modalities_config_dict: dict) -> GPT2LLM:
    model = get_model_from_config(config=modalities_config_dict, model_type=ModelTypeEnum.MODEL)
    assert isinstance(model, GPT2LLM)
    return model


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture()
def modalities_config_dict(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture()
def config_file_path(config_file_name: str) -> Path:
    config_file_path = _ROOT_DIR / Path("tests/conversion/test_configs/" + config_file_name)
    return config_file_path


@pytest.fixture(params=["gpt2_config_test.yaml"])
def config_file_name(request) -> str:
    return request.param
