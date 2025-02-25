import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from modalities.config.config import load_app_config_dict
from modalities.conversion.gpt2.modeling_gpt2 import GPT2DecoderLayer, GPT2ForCausalLM
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2Block
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
