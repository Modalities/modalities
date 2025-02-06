import json
import pickle
import tempfile
import warnings
from pathlib import Path

import pytest

from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from tests.conftest import DataPathCollection


def create_dummy_data(tmpdir_path: Path, byte_content: bytes) -> Path:
    random_temp_filename = Path(tempfile.NamedTemporaryFile(suffix=".txt").name).name
    dummy_data_path = Path(tmpdir_path, random_temp_filename)
    dummy_data_path.write_bytes(byte_content)
    return dummy_data_path


@pytest.mark.parametrize(
    "dummy_binary_content",
    [  #  note the two bytes "\xc3\xb8" correspond to the character 'ø' when interpreted as utf-8
        b"\xc3\xb8 This is \na dummy text\nwith newline chars\nand other\\n rand\xc3\xb8m\nchars.\n"
        b"It also includes malformatted json chars, like\n{{\n",
        b"This is \na dummy text\nwith newline chars\nand other\\n rand\xc3\xb8m\nchars.\n"
        b"It also includes malformatted json chars, like\n{{\nbut does not come with a trailing newline char...",
    ],
)
def test_index_creation(tmpdir: Path, dummy_binary_content: bytes):
    # dumps the dummy content to a file
    # e.g. the line "ø This is \na du"  is represented by the hex string:
    # c3 b8 20 54 68 69 73 20  69 73 20 0a 61 20 64 75
    # with c3 b8 corresponding to the utf-8 encoding of the character 'ø'.
    # Note that when cat the file, the console already inteprets the bytes as utf-8,
    # such that \xc3\xb8 is displayed as 'ø'.
    # As a workaround, we can print the hex stream via: cat file.bin | hexdump -C.
    plain_text_data_path = create_dummy_data(tmpdir_path=Path(tmpdir), byte_content=dummy_binary_content)

    # we interpret the binary content as utf-8 to create a jsonl file
    # \xc3\xb8 becomes 'ø' when interpreted as utf-8
    dummy_content_text = dummy_binary_content.decode("utf-8")
    # for each line in the dummy content, we create a json document with key 'text' and value of the line content
    jsonl_content_entries = [json.dumps(dict(text=s), ensure_ascii=False) for s in dummy_content_text.split("\n")]
    jsonl_content = "\n".join(jsonl_content_entries)

    jsonl_data_path = create_dummy_data(tmpdir_path=Path(tmpdir), byte_content=jsonl_content.encode("utf-8"))
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

    # generate the index
    generate_data_index_file(jsonl_data_path)

    # load the index to check if it is correct
    index = pickle.loads(dummy_dst_path.read_bytes())
    jsonl_content_binary = jsonl_content.encode("utf-8")
    # json.loads implicitly decodes a binary stream as utf-8
    loaded_data = [json.loads(jsonl_content_binary[offset : offset + length])["text"] for offset, length in index]
    assert loaded_data == dummy_content_text.split("\n")


@pytest.mark.parametrize(
    "use_sample_length_from_index",
    [True, False],
)
def test_large_file_lines_reader_text(indexed_dummy_data_path: DataPathCollection, use_sample_length_from_index: bool):
    raw_data_path = indexed_dummy_data_path.raw_data_path
    reader = LargeFileLinesReader(
        raw_data_path, use_sample_length_from_index=use_sample_length_from_index, encoding="utf-8"
    )
    assert raw_data_path.read_text().count("\n") == 12
    assert raw_data_path.read_text().rsplit("\n")[-1] == ""
    if use_sample_length_from_index:
        for item in reader:
            # make sure that we load valid json
            json.loads(item)
            assert item[-1] != "\n"
    else:
        # all samples must have a trailing "\n"-char
        # This is especially important when we sample rows to create sub datasets
        # If the last line does not have a trailing "\n"-char, then two samples can get merged into one
        for item in reader:
            assert item[-1] == "\n"
            json.loads(item[:-1])

    # content of dummy data contains trailing "\n"-char. Expected amount of samples therefore == amount of lines - 1
    assert len(reader) == 12
    assert all(map(len, reader))


@pytest.mark.parametrize(
    "use_sample_length_from_index",
    [True, False],
)
def test_large_file_lines_reader_binary_text_equivalence(
    indexed_dummy_data_path: DataPathCollection, use_sample_length_from_index: bool
):
    raw_data_path = indexed_dummy_data_path.raw_data_path
    reader_binary = LargeFileLinesReader(
        raw_data_path, use_sample_length_from_index=use_sample_length_from_index, encoding=None
    )
    reader_text = LargeFileLinesReader(
        raw_data_path, use_sample_length_from_index=use_sample_length_from_index, encoding="utf-8"
    )

    for item_binary, item_text in zip(reader_binary, reader_text):
        assert item_binary.decode("utf_8") == item_text
        # make sure that when we use sample length from index, we do not have a trailing "\n"-char
        assert use_sample_length_from_index == (not item_text.endswith("\n"))


def test_large_file_lines_reader_missing_source_data(dummy_data_path: DataPathCollection):
    raw_data_path = dummy_data_path.raw_data_path
    raw_data_path.unlink(missing_ok=True)
    assert not raw_data_path.exists()
    with pytest.raises(FileNotFoundError):
        LargeFileLinesReader(raw_data_path, dummy_data_path.index_path)
