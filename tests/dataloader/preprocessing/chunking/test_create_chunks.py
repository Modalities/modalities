import copy

import numpy as np
import pytest
from transformers import BatchEncoding

from modalities.dataloader.preprocessing.chunking.create_chunks import Chunking


class MockedPackedMemMapDatasetBase:
    def __init__(self, dataset: list[list[int]]):
        self.dataset = dataset
        self.sample_key = "sample_key"

    def __getitem__(self, item: int | slice) -> dict[str, list[int]]:
        return BatchEncoding({self.sample_key: self.dataset[item]})

    def __len__(self) -> int:
        return len(self.dataset)


def get_dataset(num_rows: int) -> MockedPackedMemMapDatasetBase:
    num_tokens_per_row = 5
    dataset = np.arange(num_rows).repeat(num_tokens_per_row).reshape(num_rows, num_tokens_per_row)
    dataset[:, -1] = num_rows
    dataset_list = dataset.tolist()

    mocked_dataset = MockedPackedMemMapDatasetBase(dataset_list)
    return mocked_dataset


@pytest.mark.parametrize(
    "num_chunks, num_samples, expected_chunk_indices",
    [
        (5, 10, [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]),  # num_chunks << num_samples (2 samples per chunk exactly)
        (
            5,
            8,
            [[0, 2], [2, 4], [4, 6], [6, 7], [7, 8]],
        ),  # num_chunks << num_samples (2 samples per chunk only for the first 3 chunks, 1 otherwise)
        (1, 10, [[0, 10]]),  # num_chunks == num_samples (all samples in one chunk)
        (0, 10, []),  # num_chunks == 0, no chunks
        (
            20,
            10,
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
            ],
        ),  # num_chunks >> num_samples (1 sample per chunk only for the first 10 chunks, 0 otherwise)
        (
            15,
            6,
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 6],
                [6, 6],
                [6, 6],
                [6, 6],
                [6, 6],
                [6, 6],
                [6, 6],
                [6, 6],
                [6, 6],
            ],
        ),  # num_chunks >> num_samples (1 sample per chunk only for the first 6 chunks, 0 otherwise)
        (
            10,
            1,
            [[0, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        ),  # num_chunks >> num_samples (1 sample per chunk only for the first chunk, 0 otherwise)
        (0, 10, []),  # num_chunks == 0, no chunks
    ],
)
def test__get_chunk_range(num_chunks: int, num_samples: int, expected_chunk_indices: list[list[int]]):
    chunk_indices = [
        Chunking._get_chunk_range(num_chunks=num_chunks, num_samples=num_samples, chunk_id=chunk_id)
        for chunk_id in range(num_chunks)
    ]
    assert chunk_indices == expected_chunk_indices


@pytest.mark.parametrize(
    "dataset, num_chunks, chunk_id, expect_error",
    [
        (get_dataset(10), 2, 0, False),  # num_chunks << num samples, chunk_id = 0
        (get_dataset(10), 20, 0, False),  # num_chunks >> num samples, chunk_id = 0
        (get_dataset(10), 2, 1, False),  # num_chunks << num samples, chunk_id = 1
        (get_dataset(10), 20, 1, False),  # num_chunks >> num samples, chunk_id = 1
        (get_dataset(10), 0, 0, True),  # num_chunks == 0, chunk_id = 0, -> expect error
        (get_dataset(10), 0, 1, True),  # num_chunks == 0, chunk_id = 1, -> expect error
        (get_dataset(10), 1, 0, False),  # num_chunks == 1, chunk_id = 0
        (get_dataset(10), 1, 1, True),  # num_chunks == 1, chunk_id = 1, -> expect error
        (get_dataset(0), 5, 0, False),  # empty dataset, chunk_id = 0
        (get_dataset(0), 5, 1, False),  # empty dataset, chunk_id = 1
    ],
)
def test_get_file_chunk(dataset: list[list[int]], num_chunks: int, chunk_id: int, expect_error: bool):
    if expect_error:
        with pytest.raises(ValueError):
            Chunking.get_file_chunk(dataset, num_chunks=num_chunks, chunk_id=chunk_id)
        return
    else:
        chunk = Chunking.get_file_chunk(dataset, num_chunks=num_chunks, chunk_id=chunk_id)

    chunk_range = Chunking._get_chunk_range(num_chunks=num_chunks, num_samples=len(dataset), chunk_id=chunk_id)
    chunk_recalculated = dataset[chunk_range[0] : chunk_range[1]][dataset.sample_key]

    assert chunk == chunk_recalculated


@pytest.mark.parametrize(
    "file_chunks",
    [
        (np.arange(1000).reshape(200, -1).tolist()),  # 200 samples, 5 tokens per sample
        ([]),
    ],
)
def test_shuffle_file_chunks_in_place(file_chunks: list[list[int]]):
    file_chunks_copy = copy.deepcopy(file_chunks)

    Chunking.shuffle_file_chunks_in_place(file_chunks)
    if len(file_chunks) > 0:
        assert file_chunks_copy != file_chunks
        assert sorted(file_chunks_copy) == sorted(file_chunks)
    else:
        assert len(file_chunks) == 0
