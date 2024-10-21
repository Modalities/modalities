from pathlib import Path

import pytest
import torch

from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.models.huggingface.huggingface_model import HuggingFacePretrainedModel, HuggingFacePretrainedModelConfig
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR


def test_hf_model():
    # Create model
    config_file_path = _ROOT_DIR / Path("tests/models/hf/hf_config.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    hf_config = HuggingFacePretrainedModelConfig.model_validate(config_dict)
    model = HuggingFacePretrainedModel(**dict(hf_config))

    # Create dummy inputs
    dummy_input_text = torch.randint(0, 128000, (1, 256))
    dummy_input = dict(input_ids=dummy_input_text)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Run one training step
    optimizer.zero_grad()
    out = model(dummy_input)
    loss = out["logits"].sum()
    loss.backward()
    optimizer.step()

    # Test outputs
    assert "logits" in out
    assert out["logits"].shape == (1, 256, 128256)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_hf_model_e2e(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    # Create model
    config_file_path = _ROOT_DIR / Path("config_files/training/config_example_hf.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)

    # disable checkpointing
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_strategy"]["config"]["k"] = 0

    main = Main(config_file_path)
    main.config_dict = config_dict

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)
