from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama import LlamaConfig

from modalities.config.config import load_app_config_dict
from modalities.conversion.gpt2.configuration_gpt2 import GPT2Config
from modalities.conversion.gpt2.conversion_model import (
    _build_single_node_dcp_config,
    _check_conversion_criteria,
    _copy_weights_base_modules,
    _get_layer_norm_value,
    _get_rms_norm_value,
    _load_hf_model_for_dcp_comparison,
    _map_attention_type,
    check_converted_model,
    convert_model_checkpoint,
    convert_model_config,
)
from modalities.conversion.gpt2.modeling_gpt2 import GPT2ForCausalLM
from modalities.models.components.layer_norms import LayerNormConfig, PytorchRMSLayerNormConfig
from modalities.models.gpt2.gpt2_model import PositionTypes
from tests.conversion.gpt2.helper import check_same_weight_base_modules, check_same_weight_model

CONVERSION_CASES = [
    pytest.param("gpt2_config_test.yaml", GPT2ForCausalLM, GPT2Config, id="layer-norm-gpt2"),
    pytest.param("gpt2_rmsnorm_config_test.yaml", LlamaForCausalLM, LlamaConfig, id="rms-norm-llama"),
]


@pytest.mark.parametrize(
    ("config_file_name", "expected_model_class", "expected_config_class"),
    CONVERSION_CASES,
    indirect=["config_file_name"],
)
def test_convert_model_can_generate(gpt2_config_path: Path, expected_model_class: type, expected_config_class: type):
    modalities_config = load_app_config_dict(gpt2_config_path)
    hf_model, _ = convert_model_checkpoint(modalities_config)
    assert isinstance(hf_model, expected_model_class)
    assert isinstance(hf_model.config, expected_config_class)
    assert hf_model.can_generate()


@pytest.mark.parametrize(
    ("config_file_name", "expected_model_class", "_expected_config_class"),
    CONVERSION_CASES,
    indirect=["config_file_name"],
)
def test_convert_model_checkpoint_does_not_change_weights(
    gpt2_config_path: Path, expected_model_class: type, _expected_config_class: type
):
    modalities_config = load_app_config_dict(gpt2_config_path)
    hf_model, modalities_model = convert_model_checkpoint(modalities_config)
    assert isinstance(hf_model, expected_model_class)
    check_same_weight_model(hf_model, modalities_model)


@pytest.mark.parametrize(
    ("config_file_name", "expected_model_class", "_expected_config_class"),
    CONVERSION_CASES,
    indirect=["config_file_name"],
)
def test_convert_model_checkpoint_produces_same_logits_as_original(
    gpt2_config_path: Path, expected_model_class: type, _expected_config_class: type
):
    modalities_config = load_app_config_dict(gpt2_config_path)
    hf_model, modalities_model = convert_model_checkpoint(modalities_config)
    assert isinstance(hf_model, expected_model_class)
    vocab_size = modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]["vocab_size"]
    check_converted_model(hf_model, modalities_model, num_testruns=1, vocab_size=vocab_size)


@pytest.mark.parametrize(
    ("config_file_name", "_expected_model_class", "expected_config_class"),
    CONVERSION_CASES,
    indirect=["config_file_name"],
)
def test_convert_model_config_returns_expected_hf_config(
    gpt2_config_path: Path, _expected_model_class: type, expected_config_class: type
):
    modalities_config = load_app_config_dict(gpt2_config_path)
    hf_config = convert_model_config(modalities_config)
    model_config = modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]

    assert isinstance(hf_config, expected_config_class)
    assert hf_config.hidden_size == model_config["n_embd"]
    assert hf_config.num_hidden_layers == model_config["n_layer"]
    assert hf_config.num_attention_heads == model_config["n_head_q"]
    assert hf_config.num_key_value_heads == model_config["n_head_kv"]
    assert hf_config.rope_theta == model_config["attention_config"]["qkv_transforms"][0]["config"]["base_freq"]

    expected_eps = model_config["ffn_norm_config"]["config"]["eps"]
    if isinstance(hf_config, GPT2Config):
        assert hf_config.layer_norm_eps == pytest.approx(expected_eps)
    else:
        assert hf_config.rms_norm_eps == pytest.approx(expected_eps)


@pytest.mark.parametrize("corrupt_model_head_key_in_state_dict", [True])
@pytest.mark.parametrize(
    ("config_file_name", "_expected_model_class", "_expected_config_class"),
    CONVERSION_CASES,
    indirect=["config_file_name"],
)
def test_convert_model_with_wrong_key_in_checkpoint_state_dict_fails(
    gpt2_config_path: Path,
    corrupt_model_head_key_in_state_dict: bool,
    _expected_model_class: type,
    _expected_config_class: type,
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


def test_check_conversion_criteria_rejects_invalid_position_type():
    config = _build_minimal_conversion_criteria()
    config["poe_type"] = "rope"

    with pytest.raises(AssertionError):
        _check_conversion_criteria(config)


def test_check_conversion_criteria_rejects_invalid_activation_type():
    config = _build_minimal_conversion_criteria()
    config["activation_type"] = "gelu"

    with pytest.raises(AssertionError):
        _check_conversion_criteria(config)


def test_check_conversion_criteria_rejects_mismatched_layer_norm_settings():
    config = _build_minimal_conversion_criteria()
    config["attention_norm_config"]["config"] = {"bias": True}
    config["ffn_norm_config"]["config"] = {"bias": False}
    config["lm_head_norm_config"]["config"] = {"bias": True}

    with pytest.raises(AssertionError, match="same bias setting"):
        _check_conversion_criteria(config)


def test_check_conversion_criteria_rejects_mismatched_rms_eps():
    config = _build_minimal_conversion_criteria(norm_type="pytorch_rms_norm")
    config["attention_norm_config"]["config"] = {"eps": 1e-5}
    config["ffn_norm_config"]["config"] = {"eps": 1e-6}
    config["lm_head_norm_config"]["config"] = {"eps": 1e-5}

    with pytest.raises(AssertionError, match="same eps setting"):
        _check_conversion_criteria(config)


def test_get_layer_norm_value_returns_default_when_field_missing():
    assert _get_layer_norm_value({}, "eps") == LayerNormConfig.model_fields["eps"].default


def test_get_rms_norm_value_returns_default_when_field_missing():
    assert _get_rms_norm_value({}, "eps") == PytorchRMSLayerNormConfig.model_fields["eps"].default


def test_map_attention_type_maps_supported_values():
    assert _map_attention_type({"attention_implementation": "pytorch_flash"}) == "sdpa"
    assert _map_attention_type({"attention_implementation": "manual"}) == "eager"


def test_map_attention_type_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown or unsupported attention implementation"):
        _map_attention_type({"attention_implementation": "xformers"})


def test_build_single_node_dcp_config_preserves_optional_sections(monkeypatch: pytest.MonkeyPatch):
    dcp_config = {
        "fsdp_model": {"config": {"model": {"instance_key": "old_model"}}},
        "initialized_model": {"config": {"model": {"instance_key": "placeholder"}}},
        "model_raw": {"config": {"vocab_size": 128}},
        "settings": {"config_file_path": "original.yaml"},
        "dp_degree": 2,
        "optimizer": {"name": "adamw"},
        "lr_scheduler": {"name": "onecycle"},
        "app_state": {"component_key": "app_state", "variant_key": "raw"},
    }

    monkeypatch.setattr(
        "modalities.conversion.gpt2.conversion_model.load_dcp_config",
        lambda _path: (None, deepcopy(dcp_config)),
    )

    new_config = _build_single_node_dcp_config("/tmp/checkpoint")

    assert new_config["settings"]["config_file_path"] == "converted_dcp_config.yaml"
    assert new_config["dp_degree"] == 2
    assert new_config["optimizer"] == {"name": "adamw"}
    assert new_config["lr_scheduler"] == {"name": "onecycle"}
    assert new_config["app_state"]["variant_key"] == "dcp"
    assert new_config["app_state"]["config"]["checkpoint_dir_path"] == "/tmp/checkpoint"
    assert new_config["fsdp_model"]["config"]["model"]["instance_key"] == "model_raw"
    assert new_config["initialized_model"]["config"]["model"]["instance_key"] == "fsdp_model"


def test_load_hf_model_for_dcp_comparison_sets_attention_implementation(monkeypatch: pytest.MonkeyPatch):
    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation=None)
            self.to_calls = []

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            return self

    fake_model = FakeModel()

    monkeypatch.setattr(
        "modalities.conversion.gpt2.conversion_model.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )

    dcp_config = {
        "fsdp_model": {"config": {"mixed_precision_settings": {"param_dtype": "BF_16"}}},
        "model_raw": {"config": {"attention_implementation": "manual"}},
    }

    loaded_model = _load_hf_model_for_dcp_comparison("/tmp/model", dcp_config, "cpu")

    assert loaded_model is fake_model
    assert fake_model.config._attn_implementation == "eager"
    assert len(fake_model.to_calls) == 2


def _build_minimal_conversion_criteria(norm_type: str = "layer_norm") -> dict:
    return {
        "poe_type": PositionTypes.NOPE,
        "activation_type": "swiglu",
        "attention_implementation": "pytorch_flash",
        "attention_norm_config": {"norm_type": norm_type, "config": {}},
        "ffn_norm_config": {"norm_type": norm_type, "config": {}},
        "lm_head_norm_config": {"norm_type": norm_type, "config": {}},
    }
