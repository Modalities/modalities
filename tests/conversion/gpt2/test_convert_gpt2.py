import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from modalities.config.config import load_app_config_dict
from modalities.conversion.gpt2.convert_gpt2 import (
    _copy_weights_base_modules,
    _transfer_model_code,
    check_converted_model,
    convert_gpt2,
)
from modalities.conversion.gpt2.modeling_gpt2 import GPT2DecoderLayer, GPT2ForCausalLM
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2Block
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from tests.conftest import _ROOT_DIR


def test_converting_gpt2_does_not_change_weights(tmp_path: Path, gpt2_config_path: str):
    output_dir = tmp_path / "output"
    convert_gpt2(gpt2_config_path, output_dir)
    modalities_config = load_app_config_dict(gpt2_config_path)
    original_model = get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
    converted_model = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)
    check_same_weight_model(converted_model, original_model)


def test_converting_gpt2_does_not_change_outputs(tmp_path: Path, gpt2_config_path: str):
    output_dir = tmp_path / "output"
    convert_gpt2(gpt2_config_path, output_dir)
    modalities_config = load_app_config_dict(gpt2_config_path)
    original_model = get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
    converted_model = AutoModelForCausalLM.from_pretrained(
        output_dir, local_files_only=True, trust_remote_code=True
    ).to(dtype=torch.bfloat16)
    vocab_size = modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]["vocab_size"]
    check_converted_model(converted_model, original_model, 1, vocab_size)


def test_copying_base_modules_weights_yields_identical_modules():
    m1 = nn.Linear(10, 10, bias=True)
    m2 = nn.Linear(10, 10, bias=True)
    m2.weight.data = torch.randn(10, 10)
    m2.bias.data = torch.randn(10)

    _copy_weights_base_modules(m1, m2)

    assert torch.equal(m1.weight.data, m2.weight.data)
    assert torch.equal(m1.bias.data, m2.bias.data)


def test_copying_base_modules_works_when_bias_is_false():
    m1 = nn.Linear(10, 10, bias=False)
    m2 = nn.Linear(10, 10, bias=False)
    m2.weight.data = torch.randn(10, 10)

    _copy_weights_base_modules(m1, m2)

    assert torch.equal(m1.weight.data, m2.weight.data)
    assert m1.bias == m2.bias and m2.bias is None


def test_copying_base_modules_fails_if_bias_settings_mismatch():
    m1 = nn.Linear(10, 10, bias=False)
    m2 = nn.Linear(10, 10, bias=True)
    m2.weight.data = torch.randn(10, 10)
    m2.bias.data = torch.randn(10)

    with pytest.raises(AttributeError):
        _copy_weights_base_modules(m1, m2)


def test_modeling_gpt2_gets_transferred_with_model_files(tmp_path: Path):
    modeling_gpt2_path = tmp_path / "modeling_gpt2.py"
    assert not modeling_gpt2_path.exists()
    _transfer_model_code(tmp_path)
    assert modeling_gpt2_path.exists()


def test_configuration_gpt2_gets_transferred_with_model_files(tmp_path: Path):
    configuration_gpt2_path = tmp_path / "configuration_gpt2.py"
    assert not configuration_gpt2_path.exists()
    _transfer_model_code(tmp_path)
    assert configuration_gpt2_path.exists()


def test_transferred_modeling_gpt2_does_not_import_from_modalities(tmp_path: Path):
    _transfer_model_code(tmp_path)
    with open(tmp_path / "modeling_gpt2.py") as f:
        text = f.read()
        assert "from modalities" not in text
        assert "import modalities" not in text


def test_transferred_configuration_gpt2_does_not_import_from_modalities(tmp_path: Path):
    _transfer_model_code(tmp_path)
    with open(tmp_path / "configuration_gpt2.py") as f:
        text = f.read()
        assert "from modalities" not in text
        assert "import modalities" not in text


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
def initialized_model(set_env, config_dict: dict) -> GPT2LLM:
    model = get_model_from_config(config=config_dict, model_type=ModelTypeEnum.MODEL)
    assert isinstance(model, GPT2LLM)
    return model


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture()
def config_dict(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture()
def config_file_path(config_file_name: str) -> Path:
    config_file_path = _ROOT_DIR / Path("tests/conversion/test_configs/" + config_file_name)
    return config_file_path


@pytest.fixture(params=["gpt2_config_test.yaml"])
def config_file_name(request) -> str:
    return request.param


@pytest.fixture
def device() -> str:
    return "cpu"


def check_same_weight_model(converted_model: GPT2ForCausalLM, modalities_model: GPT2LLM):
    converted_model.to(device=modalities_model.transformer.h[0].attn.q_attn.weight.device)
    assert torch.equal(converted_model.model.embed_tokens.weight, modalities_model.transformer.wte.weight)
    for i, (llama_layer, modalities_layer) in enumerate(
        zip(converted_model.model.layers, modalities_model.transformer.h)
    ):
        check_same_weight_attention(llama_layer, modalities_layer)
        check_same_weight_mlp(llama_layer, modalities_layer)
        check_same_weight_layer_norms(llama_layer, modalities_layer)
    check_same_weight_base_modules(converted_model.lm_head, modalities_model.lm_head)
    check_same_weight_base_modules(converted_model.model.norm, modalities_model.transformer.lm_head_norm)


def check_same_weight_attention(llama_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    check_same_weight_base_modules(llama_layer.self_attn.q_proj, modalities_layer.attn.q_attn)
    check_same_weight_base_modules(llama_layer.self_attn.k_proj, modalities_layer.attn.k_attn)
    check_same_weight_base_modules(llama_layer.self_attn.v_proj, modalities_layer.attn.v_attn)
    check_same_weight_base_modules(llama_layer.self_attn.o_proj, modalities_layer.attn.c_proj)


def check_same_weight_mlp(llama_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    check_same_weight_base_modules(llama_layer.mlp.down_proj, modalities_layer.mlp.W_2)
    check_same_weight_base_modules(llama_layer.mlp.gate_proj, modalities_layer.mlp.W)
    check_same_weight_base_modules(llama_layer.mlp.up_proj, modalities_layer.mlp.V)


def check_same_weight_layer_norms(llama_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    check_same_weight_base_modules(llama_layer.input_layernorm, modalities_layer.attention_norm)
    check_same_weight_base_modules(llama_layer.post_attention_layernorm, modalities_layer.ffn_norm)


def check_same_weight_base_modules(l1: nn.Linear | nn.LayerNorm, l2: nn.Linear | nn.LayerNorm):
    assert torch.equal(l1.weight, l2.weight)
    assert (l1.bias is None and l2.bias is None) or torch.equal(l1.bias, l2.bias)
    assert (l1.bias is None and l2.bias is None) or torch.equal(l1.bias, l2.bias)
    assert (l1.bias is None and l2.bias is None) or torch.equal(l1.bias, l2.bias)
