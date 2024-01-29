
import os
import pytest
from transformers import AutoConfig, AutoModelForCausalLM
import torch

from modalities.models.gpt2.pretrained_gpt_model import PretrainedGPTModel
from modalities.config.config import PretrainedGPTConfig
from modalities.__main__ import Main, load_app_config_dict


@pytest.fixture
def checkpoint_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkpoint", "checkpoint_for_testing")


@pytest.fixture
def config_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, "model_config.yaml")


@pytest.fixture
def config(config_path):
    return load_app_config_dict(config_path)


def test_convert_to_hf_checkpoint(tmp_path, config):
    # load test checkpoint
    main = Main(config)
    wrapped_model = main._get_model_from_checkpoint()

    main.load_and_convert_checkpoint(tmp_path)

    # register config and model
    AutoConfig.register("modalities_gpt2", PretrainedGPTConfig)
    AutoModelForCausalLM.register(PretrainedGPTConfig, PretrainedGPTModel)

    # load saved model
    loaded_model = AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
    loaded_model.eval()
    wrapped_model.eval()

    # check that model before and after loading return the same output
    test_tensor = torch.randint(10, size=(5, 10))

    loaded_model = loaded_model.to(wrapped_model.compute_device)
    test_tensor = test_tensor.to(wrapped_model.compute_device)

    output_before_loading = wrapped_model.forward({"input_ids": test_tensor})
    output_after_loading = loaded_model.forward(test_tensor)
    assert (output_after_loading == output_before_loading['logits']).all()