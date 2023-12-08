from unittest.mock import MagicMock

from llm_gym.__main__ import Main
from llm_gym.checkpointing.checkpointing import CheckpointingIF


def test_e2e_training_run_wout_ckpt(monkeypatch, indexed_dummy_data_path, dummy_config):
    # patch in env variables
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    dummy_config.training.train_dataloader.config.dataset.config.raw_data_path = indexed_dummy_data_path.raw_data_path
    main = Main(dummy_config)
    mocked_checkpointing = MagicMock(spec=CheckpointingIF)
    main.gym.checkpointing = mocked_checkpointing
    main.run()
    mocked_checkpointing.run.assert_called_once()
