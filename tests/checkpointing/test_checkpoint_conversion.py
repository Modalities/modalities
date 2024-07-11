import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from modalities.checkpointing.checkpoint_conversion import CheckpointConversion
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapter, HFModelAdapterConfig
from modalities.models.model import NNModel
from modalities.models.utils import ModelTypeEnum, get_model_from_config
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
def config_file_path(config_file_name: str) -> Path:
    config_file_path = _ROOT_DIR / Path("tests/checkpointing/configs_for_testing/" + config_file_name)
    return config_file_path


@pytest.fixture()
def config_dict(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture()
def initialized_model(set_env, config_dict: dict) -> NNModel:
    return get_model_from_config(config=config_dict, model_type=ModelTypeEnum.MODEL)


@pytest.fixture()
def checkpoint_conversion(tmp_path: Path, initialized_model: NNModel, config_file_path: Path) -> CheckpointConversion:
    model_file_path = tmp_path / "pytorch_model.bin"
    torch.save(initialized_model.state_dict(), model_file_path)

    output_hf_checkpoint_dir = tmp_path / "converted_hf_checkpoint"
    checkpoint_conversion = CheckpointConversion(
        config_file_path=config_file_path,
        output_hf_checkpoint_dir=output_hf_checkpoint_dir,
    )

    # Adding the checkpoint path in tmp folder to the config dict
    checkpoint_conversion.config_dict["checkpointed_model"]["config"]["checkpoint_path"] = model_file_path
    return checkpoint_conversion


@pytest.fixture()
def pytorch_model(checkpoint_conversion: CheckpointConversion) -> NNModel:
    return get_model_from_config(config=checkpoint_conversion.config_dict, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)


@pytest.fixture()
def hf_model(checkpoint_conversion: CheckpointConversion, prediction_key: str) -> NNModel:
    return checkpoint_conversion.convert_pytorch_to_hf_checkpoint(prediction_key=prediction_key)


@pytest.fixture()
def prediction_key() -> str:
    return "logits"


@pytest.fixture()
def hf_model_from_checkpoint(
    checkpoint_conversion: CheckpointConversion, pytorch_model: NNModel, device: str, prediction_key: str
) -> NNModel:
    AutoConfig.register(model_type="modalities", config=HFModelAdapterConfig)
    AutoModelForCausalLM.register(config_class=HFModelAdapterConfig, model_class=HFModelAdapter)
    hf_model_from_checkpoint = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=checkpoint_conversion.output_hf_checkpoint_dir,
        torch_dtype=pytorch_model.lm_head.weight.dtype,
        prediction_key=prediction_key,
    )
    hf_model_from_checkpoint = hf_model_from_checkpoint.to(device)
    return hf_model_from_checkpoint


@pytest.fixture()
def test_tensor(device: str, size: int = 10) -> torch.Tensor:
    test_tensor = torch.randint(size, size=(5, size))
    test_tensor = test_tensor.to(device)
    return test_tensor


def test_models_before_and_after_conversion_produce_same_output(
    device: str,
    pytorch_model: NNModel,
    hf_model: NNModel,
    hf_model_from_checkpoint: NNModel,
    test_tensor: torch.Tensor,
):
    pytorch_model = put_model_to_eval_mode(model=pytorch_model, device=device)
    hf_model = put_model_to_eval_mode(model=hf_model, device=device)

    output_pytorch_model = pytorch_model.forward(inputs={"input_ids": test_tensor})["logits"]
    output_hf_model = hf_model.forward(input_ids=test_tensor, return_dict=False)
    output_hf_model_from_checkpoint = hf_model_from_checkpoint.forward(input_ids=test_tensor, return_dict=False)

    assert (output_hf_model == output_pytorch_model).all()
    assert (output_hf_model == output_hf_model_from_checkpoint).all()


def put_model_to_eval_mode(model: NNModel, device: str) -> NNModel:
    model.eval()
    model = model.to(device)
    return model


def test_models_before_and_after_conversion_are_equal(
    pytorch_model: NNModel,
    hf_model: NNModel,
    hf_model_from_checkpoint: NNModel,
):
    for p1, p2, p3 in zip(hf_model.parameters(), pytorch_model.parameters(), hf_model_from_checkpoint.parameters()):
        assert torch.equal(p1, p2)
        assert torch.equal(p1, p3)
