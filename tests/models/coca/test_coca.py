from pathlib import Path

import pytest
import torch

from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.models.coca.coca_model import CoCa, CoCaConfig
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR


def test_coca():
    # Create model
    config_file_path = _ROOT_DIR / Path("tests/models/coca/coca_config.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    coca_config = CoCaConfig.model_validate(config_dict)
    model = CoCa(**dict(coca_config))

    # Create dummy inputs
    dummy_input_image = torch.randn(1, 3, 224, 224)
    dummy_input_text = torch.randint(
        0, coca_config.text_decoder_config.vocab_size, (1, coca_config.text_decoder_config.block_size)
    )
    dummy_input = dict(images=dummy_input_image, input_ids=dummy_input_text)

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
    assert "vision_cls" in out
    assert "text_cls" in out
    assert out["logits"].shape == (1, 1024, 50304)
    assert out["vision_cls"].shape == (1, 1, 768)
    assert out["text_cls"].shape == (1, 1, 768)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_e2e_coca_training_run_without_checkpoint(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    # Load config
    dummy_config_path = _ROOT_DIR / Path("config_files/training/config_example_coca.yaml")
    config_dict = load_app_config_dict(dummy_config_path)

    # Disable checkpointing
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_strategy"]["config"]["k"] = 0

    main = Main(dummy_config_path)
    main.config_dict = config_dict

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)
