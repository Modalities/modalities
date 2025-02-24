import copy
import json
import tempfile
from pathlib import Path

import pytest

from modalities.api import FileExistencePolicy, shuffle_jsonl_data


@pytest.fixture
def data_rows() -> list[dict]:
    return [
        {"file_id": "file_0.jsonl", "doc_id": "0"},
        {"file_id": "file_0.jsonl", "doc_id": "1"},
        {"file_id": "file_0.jsonl", "doc_id": "2"},
        {"file_id": "file_0.jsonl", "doc_id": "3"},
        {"file_id": "file_0.jsonl", "doc_id": "4"},
        {"file_id": "file_0.jsonl", "doc_id": "5"},
        {"file_id": "file_0.jsonl", "doc_id": "6"},
        {"file_id": "file_0.jsonl", "doc_id": "7"},
        {"file_id": "file_0.jsonl", "doc_id": "8"},
    ]


@pytest.fixture
def input_data_path(data_rows: list[dict], tmp_path) -> Path:
    with open(tmp_path / "input.jsonl", "w", encoding="utf-8") as f:
        for row in data_rows:
            json.dump(row, f)
            f.write("\n")
            f.flush()
    return Path(f.name)


@pytest.mark.parametrize(
    "output_data_folder_path, file_existence_policy, seed",
    [
        (Path(tempfile.mkdtemp()), FileExistencePolicy.ERROR, 42),
    ],
)
def test_shuffle_jsonl_data(
    data_rows: list[dict],
    input_data_path: Path,
    output_data_folder_path: Path,
    file_existence_policy: FileExistencePolicy,
    seed: int,
):
    data_rows_copy = copy.deepcopy(data_rows)
    output_data_path = output_data_folder_path / "output.jsonl"
    shuffle_jsonl_data(
        input_data_path=input_data_path,
        output_data_path=output_data_path,
        file_existence_policy=file_existence_policy,
        seed=seed,
    )

    with output_data_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    rows_dict_shuffled = [json.loads(line) for line in lines]

    # Check that the shuffled data contains the same rows as the input data
    assert len(data_rows) > 0
    assert len(data_rows) == len(rows_dict_shuffled)
    assert any([row != row_shuffled for row, row_shuffled in zip(data_rows_copy, rows_dict_shuffled)])
    assert set([json.dumps(d) for d in data_rows]) == set([json.dumps(d) for d in rows_dict_shuffled])
