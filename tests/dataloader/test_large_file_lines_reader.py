import json
import pickle
import tempfile
from pathlib import Path

import pytest

from llm_gym.dataloader.create_index import IndexGenerator
from llm_gym.dataloader.large_file_lines_reader import LargeFileLinesReader


def create_dummy_data(tmpdir_path: Path, content: str) -> Path:
    random_temp_filename = Path(tempfile.NamedTemporaryFile(suffix=".txt").name).name
    dummy_data_path = Path(tmpdir_path, random_temp_filename)
    dummy_data_path.write_text(content)
    return dummy_data_path


def test_index_creation(tmpdir):
    dummy_content_text = (
        "This is \na dummy text\nwith newline chars\nand other\\n randøm\nchars.\n"
        "It also includes malformatted json chars, like\n{{\n"
    )
    plain_text_data_path = create_dummy_data(tmpdir_path=Path(tmpdir), content=dummy_content_text)
    jsonl_content_entries = [json.dumps(dict(text=s)) for s in dummy_content_text.split("\n")]
    jsonl_content = "\n".join(jsonl_content_entries)
    jsonl_data_path = create_dummy_data(tmpdir_path=Path(tmpdir), content=jsonl_content)
    dummy_dst_path = Path(tmpdir, "index.pkl")

    def generate_data_index_file(data_path: Path, **kwargs):
        indexer = IndexGenerator(data_path, **kwargs)
        dummy_dst_path.unlink(missing_ok=True)
        indexer.run(dummy_dst_path)

    with pytest.raises(ValueError):
        generate_data_index_file(plain_text_data_path)
    generate_data_index_file(plain_text_data_path, drop_faulty_entries=True)
    generate_data_index_file(jsonl_data_path)

    index = pickle.loads(dummy_dst_path.read_bytes())
    assert [
        json.loads(jsonl_content[offset : offset + length])["text"] for offset, length in index
    ] == dummy_content_text.split("\n")


def test_large_file_lines_reader(dummy_data_path):
    reader = LargeFileLinesReader(dummy_data_path, lazy_init=True)
    assert dummy_data_path.read_text().count("\n") == 2
    assert dummy_data_path.read_text().rsplit("\n")[-1] == ""
    # content of dummy data contains trailing "\n"-char. Expected amount of samples therefore == amount of lines - 1
    assert len(reader) == 2
    assert all(map(len, reader))


def test_large_file_lines_reader_lazy_index_init(tmpdir, dummy_data_path):
    index_path = Path(tmpdir, f"{dummy_data_path.stem}.idx")
    with pytest.raises(FileNotFoundError):
        LargeFileLinesReader(dummy_data_path, index_path, lazy_init=False)

    LargeFileLinesReader(dummy_data_path, index_path, lazy_init=True)
    LargeFileLinesReader(dummy_data_path, index_path, lazy_init=False)


def test_large_file_lines_reader_missing_source_data(tmpdir, dummy_data_path):
    dummy_data_path.unlink(missing_ok=True)
    assert not dummy_data_path.exists()
    index_path = Path(tmpdir, f"{dummy_data_path.stem}.idx")
    with pytest.raises(FileNotFoundError):
        LargeFileLinesReader(dummy_data_path, index_path, lazy_init=False)
