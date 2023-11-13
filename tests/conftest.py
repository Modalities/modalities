from pathlib import Path

import pytest

_ROOT_DIR = Path(__file__).parents[1]


@pytest.fixture
def dummy_config_path() -> Path:
    return Path("../../config_files/config_lorem_ipsum.yaml")


@pytest.fixture
def dummy_data_path(tmpdir) -> Path:
    source_dummy_data_path = _ROOT_DIR / Path("./data/lorem_ipsum.jsonl")
    dummy_data_path = Path(tmpdir, source_dummy_data_path.name)
    dummy_data_path.write_text(source_dummy_data_path.read_text())
    index_path = Path(tmpdir, f"{dummy_data_path.stem}.idx.pkl")
    index_path.unlink(missing_ok=True)

    return dummy_data_path
