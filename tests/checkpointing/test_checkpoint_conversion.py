import os
from pathlib import Path

import pytest
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoConfig

from modalities.checkpointing.checkpoint_conversion import CheckpointConversion
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.models.huggingface_adapters.mamba_hf_adapter import MambaHuggingFaceAdapterConfig, MambaHuggingFaceModelAdapter
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


@pytest.fixture()
def checkpoint_conversion(tmp_path, initialized_model, config_file_path):
    model_file_path = tmp_path / "pytorch_model.bin"
    torch.save(initialized_model.state_dict(), model_file_path)

    output_hf_checkpoint_dir = tmp_path / "converted_hf_checkpoint"
    checkpoint_conversion = CheckpointConversion(
        config_file_path=config_file_path,
        output_hf_checkpoint_dir=output_hf_checkpoint_dir,
    )
    checkpoint_conversion.config_dict["checkpointed_model"]["config"][
        "checkpoint_path"] = model_file_path
    return checkpoint_conversion


@pytest.fixture()
def pytorch_model(checkpoint_conversion):
    return checkpoint_conversion._setup_model()


@pytest.fixture()
def hf_model(checkpoint_conversion):
    return checkpoint_conversion.convert_pytorch_to_hf_checkpoint()


@pytest.fixture()
def hf_model_from_checkpoint(checkpoint_conversion, pytorch_model, device):
    AutoConfig.register("modalities_mamba", MambaHuggingFaceAdapterConfig)
    AutoModelForCausalLM.register(MambaHuggingFaceAdapterConfig, MambaHuggingFaceModelAdapter)
    hf_model_from_checkpoint = AutoModelForCausalLM.from_pretrained(checkpoint_conversion.output_hf_checkpoint_dir,
                                                                    torch_dtype=pytorch_model.lm_head.weight.dtype)
    hf_model_from_checkpoint = hf_model_from_checkpoint.to(device)
    return hf_model_from_checkpoint


@pytest.fixture()
def test_tensor(device, size: int = 10):
    test_tensor = torch.randint(size, size=(5, size))
    test_tensor = test_tensor.to(device)
    return test_tensor


def test_hf_and_pytorch_models_are_the_same_after_init(hf_model, pytorch_model, checkpoint_conversion):
    assert hf_model.dtype == pytorch_model.lm_head.weight.dtype
    assert hf_model.__class__.__name__ == "MambaHuggingFaceModelAdapter"
    assert os.listdir(checkpoint_conversion.output_hf_checkpoint_dir)


def test_models_before_and_after_conversion_produce_same_output(
        device,
        pytorch_model,
        hf_model,
        hf_model_from_checkpoint,
        test_tensor,
):
    pytorch_model = put_model_to_eval_mode(pytorch_model, device)
    hf_model = put_model_to_eval_mode(hf_model, device)

    output_pytorch_model = pytorch_model.forward({"input_ids": test_tensor})["logits"]
    output_hf_model = hf_model.forward(test_tensor)
    output_hf_model_from_checkpoint = hf_model_from_checkpoint.forward(test_tensor)

    assert (output_hf_model == output_pytorch_model).all()
    assert (output_hf_model == output_hf_model_from_checkpoint).all()


def put_model_to_eval_mode(model, device):
    model.eval()
    model = model.to(device)
    return model
