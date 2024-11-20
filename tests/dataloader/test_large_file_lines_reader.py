import json
import pickle
import tempfile
import warnings
from pathlib import Path

import pytest

from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader, OpenFileMode


def create_dummy_data(tmpdir_path: Path, content: str) -> Path:
    random_temp_filename = Path(tempfile.NamedTemporaryFile(suffix=".txt").name).name
    dummy_data_path = Path(tmpdir_path, random_temp_filename)
    dummy_data_path.write_text(content)
    return dummy_data_path


@pytest.mark.parametrize(
    "dummy_content_text",
    [
        "This is \na dummy text\nwith newline chars\nand other\\n randøm\nchars.\n"
        "It also includes malformatted json chars, like\n{{\n",
        "This is \na dummy text\nwith newline chars\nand other\\n randøm\nchars.\n"
        "It also includes malformatted json chars, like\n{{\nbut does not come with a trailing newline char...",
    ],
)
def test_index_creation(tmpdir, dummy_content_text):
    plain_text_data_path = create_dummy_data(tmpdir_path=Path(tmpdir), content=dummy_content_text)
    jsonl_content_entries = [json.dumps(dict(text=s)) for s in dummy_content_text.split("\n")]
    jsonl_content = "\n".join(jsonl_content_entries)
    jsonl_data_path = create_dummy_data(tmpdir_path=Path(tmpdir), content=jsonl_content)
    dummy_dst_path = Path(tmpdir, "index.pkl")

    def generate_data_index_file(data_path: Path, **kwargs):
        indexer = IndexGenerator(data_path, **kwargs)
        dummy_dst_path.unlink(missing_ok=True)
        indexer.create_index(dummy_dst_path)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with pytest.raises(ValueError):
            generate_data_index_file(plain_text_data_path)
        generate_data_index_file(plain_text_data_path, drop_faulty_entries=True)
    generate_data_index_file(jsonl_data_path)

    index = pickle.loads(dummy_dst_path.read_bytes())
    assert [
        json.loads(jsonl_content[offset : offset + length])["text"] for offset, length in index
    ] == dummy_content_text.split("\n")


@pytest.mark.parametrize(
    "use_sample_length_from_index",
    [True, False],
)
def test_large_file_lines_reader_text(indexed_dummy_data_path, use_sample_length_from_index):
    raw_data_path = indexed_dummy_data_path.raw_data_path
    reader = LargeFileLinesReader(
        raw_data_path, open_file_mode=OpenFileMode.READ_TEXT, use_sample_length_from_index=use_sample_length_from_index
    )
    assert raw_data_path.read_text().count("\n") == 12
    assert raw_data_path.read_text().rsplit("\n")[-1] == ""
    if use_sample_length_from_index:
        print([item[-1] for item in reader])
        assert all([item[-1] != "\n" for item in reader])
    else:
        # all but the last sample should have a trailing "\n"-char
        # Since we don't know if the last document of  every JSONL file a trailing "\n"-char
        # we always read the last sample as defined by sample's length in the index (i.e., without any \n chars).
        assert all([item[-1] == "\n" for item in reader][:-1])
    # content of dummy data contains trailing "\n"-char. Expected amount of samples therefore == amount of lines - 1
    assert len(reader) == 12
    assert all(map(len, reader))


def test_large_file_lines_reader_binary_text_equivalence(indexed_dummy_data_path):
    raw_data_path = indexed_dummy_data_path.raw_data_path
    reader_binary = LargeFileLinesReader(
        raw_data_path, open_file_mode=OpenFileMode.READ_BINARY, use_sample_length_from_index=False
    )
    reader_text = LargeFileLinesReader(
        raw_data_path, open_file_mode=OpenFileMode.READ_TEXT, use_sample_length_from_index=False
    )
    for item_binary, item_text in zip(reader_binary, reader_text):
        assert item_binary.decode("utf_8") == item_text


def test_large_file_lines_reader_missing_source_data(tmpdir, dummy_data_path):
    raw_data_path = dummy_data_path.raw_data_path
    raw_data_path.unlink(missing_ok=True)
    assert not raw_data_path.exists()
    with pytest.raises(FileNotFoundError):
        LargeFileLinesReader(raw_data_path, dummy_data_path.index_path)
