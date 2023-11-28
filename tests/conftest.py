from pathlib import Path

import pytest

from llm_gym.__main__ import load_app_config_dict
from llm_gym.config.config import AppConfig

_ROOT_DIR = Path(__file__).parents[1]


@pytest.fixture
def dummy_config(monkeypatch) -> AppConfig:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dummy_config_path = _ROOT_DIR / Path("config_files/config_lorem_ipsum.yaml")
    config_dict = load_app_config_dict(dummy_config_path)
    return AppConfig.model_validate(config_dict)


@pytest.fixture
def dummy_data_path(tmpdir) -> Path:
    source_raw_dummy_data_path = _ROOT_DIR / Path("./data/lorem_ipsum.jsonl")
    dummy_data_path = Path(tmpdir, source_raw_dummy_data_path.name)
    dummy_data_path.write_text(source_raw_dummy_data_path.read_text())
    index_path = Path(tmpdir, f"{dummy_data_path.stem}.idx.pkl")
    index_path.unlink(missing_ok=True)
    return dummy_data_path
