import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import AppConfig, HuggingFaceModelConfig
from modalities.models.gpt2.huggingface_model import HuggingFaceModel
from tests.conftest import torch_distributed_cleanup


@pytest.fixture
def checkpoint_dir():
    return Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkpoint", "checkpoint_for_testing"))


@pytest.fixture
def config_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, "model_config.yaml")


@pytest.fixture
def config(config_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    config_dict = load_app_config_dict(config_path)
    pydantic_config = AppConfig.model_validate(config_dict)
    return pydantic_config


@pytest.fixture
def device():
    return "cpu"


def test_convert_to_hf_checkpoint(tmp_path, config, checkpoint_dir, device):
    # load test checkpoint
    main = Main(config)
    pytorch_model = main.load_and_convert_checkpoint(checkpoint_dir, tmp_path)
    # register config and model
    AutoConfig.register("modalities_gpt2", HuggingFaceModelConfig)
    AutoModelForCausalLM.register(HuggingFaceModelConfig, HuggingFaceModel)
    pytorch_model.eval()
    pytorch_model = pytorch_model.to(device)
    # check that model before and after loading return the same output
    test_tensor = torch.randint(10, size=(5, 10))
    test_tensor = test_tensor.to(device)
    output_before_loading = pytorch_model.forward({"input_ids": test_tensor})["logits"]
    # load saved model
    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype=pytorch_model.lm_head.weight.dtype)
    hf_model = hf_model.to(device)
    assert hf_model.dtype == pytorch_model.lm_head.weight.dtype
    hf_model.eval()
    output_after_loading = hf_model.forward(test_tensor)
    assert (output_after_loading == output_before_loading).all()
