import pickle
from pathlib import Path

import pytest

from .create_index import IndexGenerator
from .large_file_lines_reader import LargeFileLinesReader

_ROOT_DIR = Path(__file__).parent.parent.parent


def test_index_creation(tmpdir):
    larger_test_data_path = Path("/home/shared/openwebtext/head20000_openwebtext2_en.jsonl")
    indexer = IndexGenerator(larger_test_data_path)
    dummy_dst_path = Path(tmpdir, "index.pkl")
    indexer.run(dummy_dst_path)

    index = pickle.loads(dummy_dst_path.read_bytes())
    assert index[:5] == [(0, 477), (477, 3798), (4275, 1731), (6006, 11181), (17187, 4887)]


def test_large_file_lines_reader(tmpdir):
    source_dummy_data_path = _ROOT_DIR / Path("data/lorem_ipsum.txt")
    dummy_data_path = Path(tmpdir, source_dummy_data_path.name)
    dummy_data_path.write_text(source_dummy_data_path.read_text())
    index_path = Path(tmpdir, f"{dummy_data_path.stem}.idx.pkl")
    reader = LargeFileLinesReader(dummy_data_path, index_path, lazy_init=True)

    assert len(reader) == 3
    assert len(reader[0]) >= 0
    assert len(reader[-1]) == 0


def test_large_file_lines_reader_lazy_index_init(tmpdir):
    source_dummy_data_path = _ROOT_DIR / Path("data/lorem_ipsum.txt")
    dummy_data_path = Path(tmpdir, source_dummy_data_path.name)
    dummy_data_path.write_text(source_dummy_data_path.read_text())
    index_path = Path(tmpdir, f"{dummy_data_path.stem}.idx.pkl")
    with pytest.raises(FileNotFoundError):
        LargeFileLinesReader(dummy_data_path, index_path, lazy_init=False)

    LargeFileLinesReader(dummy_data_path, index_path, lazy_init=True)
    LargeFileLinesReader(dummy_data_path, index_path, lazy_init=False)


def test_large_file_lines_reader_missing_source_data(tmpdir):
    source_dummy_data_path = _ROOT_DIR / Path("data/lorem_ipsum.txt")
    dummy_data_path = Path(tmpdir, source_dummy_data_path.name)
    assert not dummy_data_path.exists()
    index_path = Path(tmpdir, f"{dummy_data_path.stem}.idx.pkl")
    with pytest.raises(FileNotFoundError):
        LargeFileLinesReader(dummy_data_path, index_path, lazy_init=False)
