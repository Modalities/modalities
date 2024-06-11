import os
import shutil
from pathlib import Path

import pytest
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoConfig

from modalities.checkpointing.checkpoint_conversion import CheckpointConversion
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.models.huggingface.huggingface_adapter import HuggingFaceAdapterConfig, HuggingFaceModel
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture
def device():
    return "cuda:0"


@pytest.fixture()
def component_factory():
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    return component_factory


@pytest.fixture()
def config_file_path():
    return Path("configs_for_testing/mamba_config_test.yaml")


@pytest.fixture()
def config_dict(config_file_path):
    return load_app_config_dict(config_file_path)


@pytest.fixture()
def initialized_model(set_env, component_factory, config_dict):
    class ModelConfig(BaseModel):
        model: PydanticPytorchModuleType

    components = component_factory.build_components(
        config_dict=config_dict, components_model_type=ModelConfig
    )
    return components.model


def test_entry_point_convert_pytorch_to_hf_checkpoint(
        initialized_model,
        config_file_path,
        device,
        tmp_path
):
    output_hf_checkpoint_dir = tmp_path / "converted_hf_checkpoint"
    model_file_path = tmp_path / "pytorch_model.bin"

    torch.save(initialized_model.state_dict(), model_file_path)
    checkpoint_conversion = CheckpointConversion(
        config_file_path=config_file_path,
        output_hf_checkpoint_dir=output_hf_checkpoint_dir,
    )

    checkpoint_conversion.config_dict["checkpointed_model"]["config"][
        "checkpoint_path"] = model_file_path
    pytorch_model = checkpoint_conversion._setup_model()
    hf_model = checkpoint_conversion.convert_pytorch_to_hf_checkpoint()

    # register config and model
    AutoConfig.register("modalities", HuggingFaceAdapterConfig)
    AutoModelForCausalLM.register(HuggingFaceAdapterConfig, HuggingFaceModel)

    hf_model_from_checkpoint = AutoModelForCausalLM.from_pretrained(output_hf_checkpoint_dir,
                                                                    torch_dtype=pytorch_model.lm_head.weight.dtype)
    hf_model_from_checkpoint = hf_model_from_checkpoint.to(device)

    assert hf_model.dtype == pytorch_model.lm_head.weight.dtype
    assert hf_model.__class__.__name__ == "HuggingFaceModel"
    assert os.listdir(output_hf_checkpoint_dir)

    # Evaluating whether the model before and after conversion produces the same output
    pytorch_model.eval()
    pytorch_model = pytorch_model.to(device)
    hf_model.eval()
    hf_model = hf_model.to(device)

    test_tensor = torch.randint(10, size=(5, 10))
    test_tensor = test_tensor.to(device)
    output_pytorch_model = pytorch_model.forward({"input_ids": test_tensor})["logits"]

    output_hf_model = hf_model.forward(test_tensor)
    output_hf_model_from_checkpoint = hf_model_from_checkpoint.forward(test_tensor)

    assert (output_hf_model == output_pytorch_model).all()
    assert (output_hf_model == output_hf_model_from_checkpoint).all()
