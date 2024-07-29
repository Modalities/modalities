import os
from pathlib import Path

import pytest
from torch import nn

from modalities.config.config import load_app_config_dict
from modalities.models.lora.utils import convert_layer
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
        "tests/fine_tuning/test_configs/" + "config_lorem_ipsum_sft.yaml"
    )
    return config_file_path


@pytest.fixture()
def config_dict(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture()
def model(set_env, config_dict: dict) -> NNModel:
    return get_model_from_config(config=config_dict, model_type=ModelTypeEnum.MODEL)


@pytest.fixture()
def r():
    return 10


@pytest.fixture()
def alpha():
    return 1


def test_convert_linear_layer(model, r, alpha):
    layer_to_convert = model.lm_head
    assert isinstance(layer_to_convert, nn.Linear)
    lora_linear = convert_layer(layer_to_convert, r=r, alpha=alpha)
    assert (lora_linear.weight == layer_to_convert.weight).all()
    assert lora_linear.bias == layer_to_convert.bias
    assert lora_linear.r == r
    assert lora_linear.lora_alpha == alpha
    assert lora_linear.lora_A.shape[0] == r
    assert lora_linear.lora_B.shape[1] == r


def test_convert_embedding_layer(model, r, alpha):
    layer_to_convert = model.transformer.wte
    assert isinstance(layer_to_convert, nn.Embedding)
    lora_embedding = convert_layer(layer_to_convert, r=r, alpha=alpha)
    assert (lora_embedding.weight == layer_to_convert.weight).all()
    assert lora_embedding.r == r
    assert lora_embedding.lora_alpha == alpha
    assert lora_embedding.lora_A.shape[0] == r
    assert lora_embedding.lora_B.shape[1] == r
