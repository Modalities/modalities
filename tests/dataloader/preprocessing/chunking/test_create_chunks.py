import pytest

from modalities.dataloader.preprocessing.chunking.create_chunks import Chunking


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
        Chunking.get_chunk_range_(num_chunks=num_chunks, num_samples=num_samples, chunk_id=chunk_id)
        for chunk_id in range(num_chunks)
    ]
    assert chunk_indices == expected_chunk_indices
