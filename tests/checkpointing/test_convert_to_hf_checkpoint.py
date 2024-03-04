import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from modalities.config.config import GPT2HuggingFaceAdapterConfig, PydanticModelIFType, load_app_config_dict
from modalities.models.gpt2.huggingface_model import HuggingFaceModel
from modalities.checkpointing import checkpoint_conversion


@pytest.fixture
def checkpoint_dir():
    return Path(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkpoint", "checkpoint_for_testing")
    )


@pytest.fixture
def config_path(checkpoint_dir):
    return Path(os.path.join(checkpoint_dir, "model_config.yaml"))


@pytest.fixture
def device():
    return "cpu"


def test_convert_to_hf_checkpoint(tmp_path, config_path, device):
    # load test checkpoint
    cp = checkpoint_conversion.CheckpointConversion(
        checkpoint_dir=config_path.parent,
        config_file_name=config_path.name,
        model_file_name="",
        output_hf_checkpoint_dir=tmp_path
    )
    pytorch_model = cp._setup_model()
    with patch.object(
            checkpoint_conversion.CheckpointConversion,
            "_get_model_from_checkpoint",
            return_value=pytorch_model
    ):
        cp.convert_pytorch_to_hf_checkpoint()

    pytorch_model.eval()
    pytorch_model = pytorch_model.to(device)

    # check that model before and after loading return the same output
    test_tensor = torch.randint(10, size=(5, 10))
    test_tensor = test_tensor.to(device)
    output_before_loading = pytorch_model.forward({"input_ids": test_tensor})["logits"]

    # register config and model
    AutoConfig.register("modalities_gpt2", GPT2HuggingFaceAdapterConfig)
    AutoModelForCausalLM.register(GPT2HuggingFaceAdapterConfig, HuggingFaceModel)

    # load saved model
    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype=pytorch_model.lm_head.weight.dtype)
    hf_model = hf_model.to(device)

    assert hf_model.dtype == pytorch_model.lm_head.weight.dtype
    hf_model.eval()
    output_after_loading = hf_model.forward(test_tensor)
    assert (output_after_loading == output_before_loading).all()

    # Delete temporary model folder
    shutil.rmtree(tmp_path.parent)
    assert os.path.exists(tmp_path.parent) is False
