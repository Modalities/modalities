import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from modalities.api import FileExistencePolicy, create_shuffled_jsonl_dataset_chunk


@pytest.fixture
def input_data_root_path() -> Path:
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def output_chunk_file_path() -> Path:
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir) / "output_chunk.jsonl"
    shutil.rmtree(temp_dir)


@pytest.fixture
def input_file_list_path(input_data_root_path: Path) -> Path:
    data = {
        "file_0.jsonl": [
            {"file_id": "file_0.jsonl", "doc_id": "0"},
            {"file_id": "file_0.jsonl", "doc_id": "1"},
            {"file_id": "file_0.jsonl", "doc_id": "2"},
            {"file_id": "file_0.jsonl", "doc_id": "3"},
            {"file_id": "file_0.jsonl", "doc_id": "4"},
            {"file_id": "file_0.jsonl", "doc_id": "5"},
            {"file_id": "file_0.jsonl", "doc_id": "6"},
            {"file_id": "file_0.jsonl", "doc_id": "7"},
            {"file_id": "file_0.jsonl", "doc_id": "8"},
        ],
        "file_1.jsonl": [
            {"file_id": "file_1.jsonl", "doc_id": "0"},
            {"file_id": "file_1.jsonl", "doc_id": "1"},
            {"file_id": "file_1.jsonl", "doc_id": "2"},
            {"file_id": "file_1.jsonl", "doc_id": "3"},
        ],
        "file_2.jsonl": [
            {"file_id": "file_2.jsonl", "doc_id": "0"},
            {"file_id": "file_2.jsonl", "doc_id": "1"},
            {"file_id": "file_2.jsonl", "doc_id": "2"},
            {"file_id": "file_2.jsonl", "doc_id": "3"},
            {"file_id": "file_2.jsonl", "doc_id": "4"},
            {"file_id": "file_2.jsonl", "doc_id": "5"},
        ],
    }

    file_names = []
    for filename, content in data.items():
        file_names.append(input_data_root_path / filename)
        file_path = input_data_root_path / filename
        with file_path.open("w", encoding="utf-8") as f:
            for entry in content:
                json.dump(entry, f)
                f.write("\n")

    file_list_path = input_data_root_path / "file_list.txt"
    with open(file_list_path, "w") as f:
        for file_name in file_names:
            f.write(str(file_name) + "\n")
    return file_list_path


@pytest.mark.parametrize(
    "chunk_id, num_chunks, file_existence_policy, global_seed, expected_data",
    [
        (
            0,
            3,
            "error",
            1,
            [
                {"file_id": "file_0.jsonl", "doc_id": "0"},
                {"file_id": "file_0.jsonl", "doc_id": "1"},
                {"file_id": "file_0.jsonl", "doc_id": "2"},
                {"file_id": "file_1.jsonl", "doc_id": "0"},
                {"file_id": "file_1.jsonl", "doc_id": "1"},
                {"file_id": "file_2.jsonl", "doc_id": "0"},
                {"file_id": "file_2.jsonl", "doc_id": "1"},
            ],
        ),
        (
            1,
            3,
            "error",
            1,
            [
                {"file_id": "file_0.jsonl", "doc_id": "3"},
                {"file_id": "file_0.jsonl", "doc_id": "4"},
                {"file_id": "file_0.jsonl", "doc_id": "5"},
                {"file_id": "file_1.jsonl", "doc_id": "2"},
                {"file_id": "file_2.jsonl", "doc_id": "2"},
                {"file_id": "file_2.jsonl", "doc_id": "3"},
            ],
        ),
        (
            2,
            3,
            "error",
            1,
            [
                {"file_id": "file_0.jsonl", "doc_id": "6"},
                {"file_id": "file_0.jsonl", "doc_id": "7"},
                {"file_id": "file_0.jsonl", "doc_id": "8"},
                {"file_id": "file_1.jsonl", "doc_id": "3"},
                {"file_id": "file_2.jsonl", "doc_id": "4"},
                {"file_id": "file_2.jsonl", "doc_id": "5"},
            ],
        ),
    ],
)
def test_create_shuffled_jsonl_dataset_chunk(
    input_file_list_path: Path,
    input_data_root_path: Path,
    output_chunk_file_path: Path,
    chunk_id: int,
    num_chunks: int,
    file_existence_policy: FileExistencePolicy,
    global_seed: Optional[int],
    expected_data: dict,
):
    # This test verifies the correctness of the chunking and shuffling implemented in the
    # create_shuffled_jsonl_dataset_chunk API function. The function takes the input file list,
    # reads the data from the files, creates a chunk, shuffles the chunk and writes the shuffled
    # chunk to a new JSONL file. The correctness of the chunking and shuffling is verified by
    # comparing the output chunk to the expected data.

    file_existence_policy = FileExistencePolicy(file_existence_policy)

    with open(input_file_list_path, "r", encoding="utf-8") as f:
        file_path_list = f.readlines()
    file_path_list = [
        input_data_root_path / Path(file_path.strip()).with_suffix(".jsonl") for file_path in file_path_list
    ]

    create_shuffled_jsonl_dataset_chunk(
        file_path_list=file_path_list,
        output_chunk_file_path=output_chunk_file_path,
        chunk_id=chunk_id,
        num_chunks=num_chunks,
        file_existence_policy=file_existence_policy,
        global_seed=global_seed,
    )

    with output_chunk_file_path.open("r", encoding="utf-8") as f:
        content = f.readlines()

    chunk_dicts = [json.loads(line) for line in content]

    # make sure that the total number of documents within the chunk are correct
    assert len(chunk_dicts) == len(expected_data)

    # make sure that the documents appear in the chunk
    for doc in chunk_dicts:
        assert doc in expected_data

    # make sure that the chunk is shuffled
    assert any([doc != expected_doc for doc, expected_doc in zip(chunk_dicts, expected_data)])
