from unittest.mock import MagicMock

import pytest
import torch.cuda

from llm_gym.__main__ import Main
from llm_gym.checkpointing.checkpointing import CheckpointingIF


def no_gpu_available() -> bool:
    return not torch.cuda.is_available()


@pytest.mark.skipif(
    no_gpu_available(), reason="This e2e test verifies a GPU-Setup and uses components, which do not support CPU-only."
)
def test_e2e_training_run_wout_ckpt(monkeypatch, indexed_dummy_data_path, dummy_config):
    # patch in env variables
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    dummy_config.training.train_dataloader.config.dataset.config.raw_data_path = indexed_dummy_data_path.raw_data_path
    for val_dataloader_config in dummy_config.training.evaluation_dataloaders.values():
        val_dataloader_config.config.dataset.config.raw_data_path = indexed_dummy_data_path.raw_data_path
    main = Main(dummy_config)
    mocked_checkpointing = MagicMock(spec=CheckpointingIF)
    main.gym.checkpointing = mocked_checkpointing
    main.run()
    mocked_checkpointing.run.assert_called_once()
