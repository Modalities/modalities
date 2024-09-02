import copy
import os
from pathlib import Path

import pytest
from torch import nn

from modalities.config.config import load_app_config_dict
from modalities.models.gpt2.gpt2_model import CausalSelfAttention
from modalities.models.lora.lora_layers import (
    LoRALinear,
    LoRAConv1d,
    LoRAConv2d,
    LoRAConv3d,
)
from modalities.models.lora.utils import (
    convert_layer,
    convert_to_lora,
    convert_convXd,
    recursive_layer_conversion,
)
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
    config_file_path = _ROOT_DIR / Path("tests/fine_tuning/test_configs/" + "config_lorem_ipsum_lora_conversion.yaml")
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
def layers():
    return ["attn"]


def compute_trainable_num_parameters(model: nn.Module):
    trainable_num_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_num_params += param.numel()
    trainable_percentage = 100 * trainable_num_params / total_params
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


def test_conv1d(r, alpha):
    conv1d = nn.Conv1d(3, 6, 5)
    lora_conv1d = convert_convXd(conv1d, r, alpha)
    assert isinstance(lora_conv1d, LoRAConv1d)
    assert lora_conv1d.conv.weight.shape == conv1d.weight.shape


def test_conv2d(r, alpha):
    conv2d = nn.Conv2d(3, 6, 5)
    lora_conv2d = convert_convXd(conv2d, r, alpha)
    assert isinstance(lora_conv2d, LoRAConv2d)
    assert lora_conv2d.conv.weight.shape == conv2d.weight.shape


def test_conv3d(r, alpha):
    conv3d = nn.Conv3d(3, 6, 5)
    lora_conv3d = convert_convXd(conv3d, r, alpha)
    assert isinstance(lora_conv3d, LoRAConv3d)
    assert lora_conv3d.conv.weight.shape == conv3d.weight.shape


def test_incorrect_type(r, alpha):
    try:
        convert_convXd(42, r, alpha)
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_attribute_copying(r, alpha):
    conv2d = nn.Conv2d(3, 6, 5, stride=2, padding=1, dilation=2, groups=1, bias=True)
    lora_conv2d = convert_convXd(conv2d, r, alpha)
    # Check all attributes
    assert lora_conv2d.conv.in_channels == conv2d.in_channels
    assert lora_conv2d.conv.out_channels == conv2d.out_channels
    assert lora_conv2d.conv.kernel_size == conv2d.kernel_size
    assert lora_conv2d.conv.stride == conv2d.stride
    assert lora_conv2d.conv.padding == conv2d.padding
    assert lora_conv2d.conv.dilation == conv2d.dilation
    assert lora_conv2d.conv.groups == conv2d.groups
    assert lora_conv2d.conv.padding_mode == conv2d.padding_mode
    assert (lora_conv2d.conv.bias == conv2d.bias).all()


@pytest.mark.parametrize(
    "list_allowed_conversion_layers",
    [
        ["q_proj"],
        ["v_proj"],
    ],
)
def test_layer_conversion(model, r, alpha, list_allowed_conversion_layers):
    recursive_layer_conversion(model, r, alpha, list_allowed_conversion_layers)
    assert not list_allowed_conversion_layers[0] in [
        type(i).__name__ for i in model.modules()
    ], "The layer type shouldn't be present in the whole model anymore."


def test_no_layer_conversion(model, r, alpha):
    list_allowed_conversion_types = []
    model_before = copy.deepcopy(model)
    recursive_layer_conversion(model, r, alpha, list_allowed_conversion_types)
    model_after = model
    module_types_before = [type(i).__name__ for i in model_before.modules()]
    module_types_after = [type(i).__name__ for i in model_after.modules()]
    assert (
        module_types_before == module_types_after
    ), "If I don't give any layers for conversion the model should be the same after."
