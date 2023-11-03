import pickle
from pathlib import Path

import pytest

from llm_gym.dataloader.create_index import IndexGenerator
from llm_gym.dataloader.large_file_lines_reader import LargeFileLinesReader

_ROOT_DIR = Path(__file__).parent.parent.parent


def test_index_creation(tmpdir):
    dummy_data_path = Path(tmpdir, "dummy_data.txt")
    # encode this as bytes. Using strings results into index shifting, if non-ascii chars are included.
    dummy_text = "This is just\na dummy text\nwith newline chars\nand other\\n randøm\nchars...".encode("utf8")
    dummy_data_path.write_bytes(dummy_text)
    indexer = IndexGenerator(dummy_data_path)
    dummy_dst_path = Path(tmpdir, "index.pkl")
    indexer.run(dummy_dst_path)

    index = pickle.loads(dummy_dst_path.read_bytes())
    assert len(index) == dummy_text.count(b"\n") + 1
    assert dummy_text.split(b"\n") == [dummy_text[offset : offset + length] for offset, length in index]


def test_large_file_lines_reader(tmpdir):
    source_dummy_data_path = _ROOT_DIR / Path("data/lorem_ipsum.txt")
    dummy_data_path = Path(tmpdir, source_dummy_data_path.name)
    dummy_data_path.write_text(source_dummy_data_path.read_text())
    index_path = Path(tmpdir, f"{dummy_data_path.stem}.idx.pkl")
    reader = LargeFileLinesReader(dummy_data_path, index_path, lazy_init=True)

    assert dummy_data_path.read_text().count("\n") == 2
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
