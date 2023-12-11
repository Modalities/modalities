import dataclasses
import pickle
from pathlib import Path

import pytest
from transformers import GPT2TokenizerFast

from llm_gym.__main__ import load_app_config_dict
from llm_gym.config.config import AppConfig
from llm_gym.dataloader.create_index import IndexGenerator
from llm_gym.dataloader.large_file_lines_reader import LargeFileLinesReader

_ROOT_DIR = Path(__file__).parents[1]


@pytest.fixture
def dummy_packed_data_path(tmpdir) -> Path:
    data = b""
    header_size_in_bytes = 8
    int_size_in_bytes = 4
    tokens = list(range(20))
    data += (len(tokens) * int_size_in_bytes).to_bytes(header_size_in_bytes, byteorder="big")
    data += b"".join([t.to_bytes(int_size_in_bytes, byteorder="big") for t in tokens])
    index = [(4, 24), (28, 40), (68, 12), (80, 4)]  # [(index,len), ...] -> in 4 bytes #lengths: 6,10,3,1
    data += pickle.dumps(index)
    dummy_packed_data_path = Path(tmpdir, "dummy.pbin")
    dummy_packed_data_path.write_bytes(data)
    return dummy_packed_data_path


@pytest.fixture
def dummy_config(monkeypatch) -> AppConfig:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dummy_config_path = _ROOT_DIR / Path("config_files/config_lorem_ipsum.yaml")
    config_dict = load_app_config_dict(dummy_config_path)
    return AppConfig.model_validate(config_dict)


@dataclasses.dataclass
class DataPathCollection:
    raw_data_path: Path
    index_path: Path


@pytest.fixture
def dummy_data_path(tmpdir) -> DataPathCollection:
    source_raw_dummy_data_path = _ROOT_DIR / Path("./data/lorem_ipsum.jsonl")
    dummy_data_path = Path(tmpdir, source_raw_dummy_data_path.name)
    dummy_data_path.write_text(source_raw_dummy_data_path.read_text())
    index_path = LargeFileLinesReader.default_index_path(dummy_data_path, index_path=None)
    index_path.unlink(missing_ok=True)
    return DataPathCollection(raw_data_path=dummy_data_path, index_path=index_path)


@pytest.fixture
def indexed_dummy_data_path(dummy_data_path) -> DataPathCollection:
    index_generator = IndexGenerator(dummy_data_path.raw_data_path)
    index_generator.run(dummy_data_path.index_path)
    return dummy_data_path


@pytest.fixture
def gpt2_tokenizer() -> GPT2TokenizerFast:
    default_gpt2_tokenizer_path = Path(__file__).parents[1] / Path("data", "tokenizer", "tokenizer.json")
    assert default_gpt2_tokenizer_path.is_file()
    return GPT2TokenizerFast(tokenizer_file=str(default_gpt2_tokenizer_path))
