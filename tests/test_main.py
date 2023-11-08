from pathlib import Path
from unittest.mock import MagicMock

from llm_gym.__main__ import Main
from llm_gym.checkpointing.checkpointing import CheckpointingIF


def test_e2e_training_run_wout_ckpt(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")
    dummy_data_path = Path(__file__).parent.parent / Path("data", "lorem_ipsum.jsonl")
    main = Main(dataset_path=dummy_data_path, num_epochs=1)
    mocked_checkpointing = MagicMock(spec=CheckpointingIF)
    main.gym.checkpointing = mocked_checkpointing
    main.run()
    mocked_checkpointing.run.assert_called_once()
