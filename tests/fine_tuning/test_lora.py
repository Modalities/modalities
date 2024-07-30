import os
from pathlib import Path

import pytest
from torch import nn

from modalities.config.config import load_app_config_dict
from modalities.models.lora.lora_layers import LoRALinear
from modalities.models.lora.utils import convert_layer, conversion_lora
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
    config_file_path = _ROOT_DIR / Path("tests/fine_tuning/test_configs/" + "config_lorem_ipsum_sft.yaml")
    return config_file_path


@pytest.fixture()
def config_dict(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture()
def model(set_env, config_dict: dict) -> NNModel:
    return get_model_from_config(config=config_dict, model_type=ModelTypeEnum.MODEL)


@pytest.fixture()
def r():
    return 8


@pytest.fixture()
def alpha():
    return 1


@pytest.fixture()
def layer_types():
    return ["attn"]


def compute_trainable_num_parameters(model: nn.Module):
    trainable_num_params = 0

    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_num_params += param.numel()

    trainable_percentage = 100 * trainable_num_params / total_params

    print(
        f"trainable params: {trainable_num_params} || \
          total params: {total_params} || \
          trainable%: {trainable_percentage}"
    )

    return trainable_percentage


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


def test_replace_modules_in_attention(model, r, alpha):
    percentage_trainable_params_before_lora = compute_trainable_num_parameters(model=model)
    assert isinstance(model.transformer.h[0].attn.c_proj, nn.Linear)

    converted = conversion_lora(model, r, alpha)
    percentage_trainable_params_after_lora = compute_trainable_num_parameters(model=model)

    assert isinstance(converted, nn.Module)
    # Checking the percentage of trainable weights before and after conversion.
    assert percentage_trainable_params_before_lora > percentage_trainable_params_after_lora, \
        "Percentage of trainable weights should be greater before lora."

    # Checking if the conversion from nn.Linear to LoRALinear actually happened.
    assert isinstance(model.transformer.h[0].attn.c_proj,
                      LoRALinear), "After conversion nn.Linear should be a LoRALinear."
