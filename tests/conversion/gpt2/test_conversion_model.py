import pytest
import torch
import torch.nn as nn

from modalities.config.config import load_app_config_dict
from modalities.conversion.gpt2.conversion_model import (
    _copy_weights_base_modules,
    check_converted_model,
    convert_model_checkpoint,
)
from tests.conversion.gpt2.helper import check_same_weight_base_modules, check_same_weight_model


def test_convert_model_can_generate(gpt2_config_path: str):
    modalities_config = load_app_config_dict(gpt2_config_path)
    hf_model, _ = convert_model_checkpoint(modalities_config)
    assert hf_model.can_generate()


def test_convert_model_checkpoint_does_not_change_weights(gpt2_config_path: str):
    modalities_config = load_app_config_dict(gpt2_config_path)
    hf_model, modalities_model = convert_model_checkpoint(modalities_config)
    check_same_weight_model(hf_model, modalities_model)


def test_convert_model_checkpoint_produces_same_logits_as_original(gpt2_config_path: str):
    modalities_config = load_app_config_dict(gpt2_config_path)
    hf_model, modalities_model = convert_model_checkpoint(modalities_config)
    vocab_size = modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]["vocab_size"]
    check_converted_model(hf_model, modalities_model, num_testruns=1, vocab_size=vocab_size)


@pytest.mark.parametrize("corrupt_model_head_key_in_state_dict", [True])
def test_convert_model_with_wrong_key_in_checkpoint_state_dict_fails(
    gpt2_config_path: str, corrupt_model_head_key_in_state_dict: bool
):
    modalities_config = load_app_config_dict(gpt2_config_path)
    with pytest.raises(RuntimeError):
        convert_model_checkpoint(modalities_config)


def test_copying_base_modules_weights_yields_identical_modules():
    m1 = nn.Linear(10, 10, bias=True)
    m2 = nn.Linear(10, 10, bias=True)
    m2.weight.data = torch.randn(10, 10)
    m2.bias.data = torch.randn(10)

    _copy_weights_base_modules(m1, m2)

    check_same_weight_base_modules(m1, m2)


def test_copying_base_modules_works_when_bias_is_false():
    m1 = nn.Linear(10, 10, bias=False)
    m2 = nn.Linear(10, 10, bias=False)
    m2.weight.data = torch.randn(10, 10)

    _copy_weights_base_modules(m1, m2)

    check_same_weight_base_modules(m1, m2)


def test_copying_base_modules_fails_if_bias_settings_mismatch():
    m1 = nn.Linear(10, 10, bias=False)
    m2 = nn.Linear(10, 10, bias=True)
    m2.weight.data = torch.randn(10, 10)
    m2.bias.data = torch.randn(10)

    with pytest.raises(AttributeError):
        _copy_weights_base_modules(m1, m2)
