from unittest.mock import MagicMock

from llm_gym.__main__ import Main
from llm_gym.checkpointing.checkpointing import CheckpointingIF


def test_e2e_training_run_wout_ckpt(monkeypatch, lorem_ipsum_data):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")
    main = Main(dataset_path=lorem_ipsum_data, num_epochs=1)
    mocked_checkpointing = MagicMock(spec=CheckpointingIF)
    main.gym.checkpointing = mocked_checkpointing
    main.run()
    mocked_checkpointing.run.assert_called_once()
