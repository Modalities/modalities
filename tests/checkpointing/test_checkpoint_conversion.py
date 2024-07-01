import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoConfig

from modalities.checkpointing.checkpoint_conversion import CheckpointConversion
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter, HFModelAdapterConfig
from modalities.models.model import NNModel
from modalities.models.utils import get_model_from_config
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from tests.conftest import _ROOT_DIR


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture
def device() -> str:
    return "cuda:0"


@pytest.fixture()
def component_factory() -> ComponentFactory:
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    return component_factory


@pytest.fixture(params=["gpt2_config_test.yaml", "mamba_config_test.yaml"])
def config_file_name(request) -> str:
    return request.param


@pytest.fixture()
def config_file_path(config_file_name) -> Path:
    config_file_path = _ROOT_DIR / Path("tests/checkpointing/configs_for_testing/" + config_file_name)
    return config_file_path


@pytest.fixture()
def config_dict(config_file_path) -> dict:
    return load_app_config_dict(config_file_path)


@pytest.fixture()
def initialized_model(set_env, config_dict) -> NNModel:
    return get_model_from_config(config=config_dict, model_type="model")


@pytest.fixture()
def checkpoint_conversion(tmp_path, initialized_model, config_file_path) -> CheckpointConversion:
    model_file_path = tmp_path / "pytorch_model.bin"
    torch.save(initialized_model.state_dict(), model_file_path)

    output_hf_checkpoint_dir = tmp_path / "converted_hf_checkpoint"
    checkpoint_conversion = CheckpointConversion(
        config_file_path=config_file_path,
        output_hf_checkpoint_dir=output_hf_checkpoint_dir,
    )
    checkpoint_conversion.config_dict["checkpointed_model"]["config"]["checkpoint_path"] = model_file_path
    return checkpoint_conversion


@pytest.fixture()
def pytorch_model(checkpoint_conversion) -> NNModel:
    return get_model_from_config(config=checkpoint_conversion.config_dict, model_type="checkpointed_model")


@pytest.fixture()
def hf_model(checkpoint_conversion) -> NNModel:
    return checkpoint_conversion.convert_pytorch_to_hf_checkpoint()


@pytest.fixture()
def hf_model_from_checkpoint(checkpoint_conversion, pytorch_model, hf_model, device) -> NNModel:
    AutoConfig.register("modalities", HFModelAdapterConfig)
    AutoModelForCausalLM.register(HFModelAdapterConfig, HFModelAdapter)
    hf_model_from_checkpoint = AutoModelForCausalLM.from_pretrained(
        checkpoint_conversion.output_hf_checkpoint_dir, torch_dtype=pytorch_model.lm_head.weight.dtype
    )
    hf_model_from_checkpoint = hf_model_from_checkpoint.to(device)
    return hf_model_from_checkpoint


def test_models_before_and_after_conversion_are_equal(
    pytorch_model,
    hf_model,
    hf_model_from_checkpoint,
):
    for p1, p2, p3 in zip(hf_model.parameters(), pytorch_model.parameters(), hf_model_from_checkpoint.parameters()):
        assert torch.equal(p1, p2)
        assert torch.equal(p1, p3)
