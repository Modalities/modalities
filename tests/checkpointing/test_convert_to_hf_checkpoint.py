import os
import pytest
from transformers import AutoConfig, AutoModelForCausalLM
import torch

from modalities.models.gpt2.pretrained_gpt_model import PretrainedGPTModel
from modalities.config.config import PretrainedGPTConfig
from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import AppConfig
from tests.conftest import torch_distributed_cleanup


@pytest.fixture
def checkpoint_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkpoint", "checkpoint_for_testing")


@pytest.fixture
def config_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, "model_config.yaml")


@pytest.fixture
def config(config_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")
    torch_distributed_cleanup()
    config_dict = load_app_config_dict(config_path)
    pydantic_config = AppConfig.model_validate(config_dict)
    return pydantic_config


def test_convert_to_hf_checkpoint(tmp_path, config):
    # load test checkpoint
    main = Main(config)
    main.load_and_convert_checkpoint(tmp_path)
    wrapped_model = main.wrapped_model
    # register config and model
    AutoConfig.register("modalities_gpt2", PretrainedGPTConfig)
    AutoModelForCausalLM.register(PretrainedGPTConfig, PretrainedGPTModel)
    # load saved model
    loaded_model = AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype=wrapped_model.module.lm_head.weight.dtype)
    assert loaded_model.dtype == wrapped_model.module.lm_head.weight.dtype
    loaded_model.eval()
    wrapped_model.eval()
    # check that model before and after loading return the same output
    test_tensor = torch.randint(10, size=(5, 10))
    loaded_model = loaded_model.to(wrapped_model.compute_device)
    test_tensor = test_tensor.to(wrapped_model.compute_device)
    output_before_loading = wrapped_model.forward({"input_ids": test_tensor})['logits']
    output_after_loading = loaded_model.forward(test_tensor)
    assert (output_after_loading == output_before_loading).all()
