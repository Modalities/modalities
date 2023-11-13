from unittest.mock import MagicMock

from llm_gym.__main__ import Main, hydra_load_app_config_dict
from llm_gym.checkpointing.checkpointing import CheckpointingIF
from llm_gym.config.config import AppConfig


def test_e2e_training_run_wout_ckpt(monkeypatch, dummy_data_path, dummy_config_path):
    # patch in env variables
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    # load and run test config
    config_dict = hydra_load_app_config_dict(dummy_config_path)
    config_dict["data"]["dataset_dir_path"] = dummy_data_path
    config = AppConfig.model_validate(config_dict)
    main = Main(config)
    mocked_checkpointing = MagicMock(spec=CheckpointingIF)
    main.gym.checkpointing = mocked_checkpointing
    main.run()
    mocked_checkpointing.run.assert_called_once()
