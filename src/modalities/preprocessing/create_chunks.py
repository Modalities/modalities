import math
from typing import Any, Optional

import numpy as np

from modalities.dataloader.dataset import PackedMemMapDatasetBase


class Chunking:
    @staticmethod
    def _get_chunk_range(num_chunks: int, num_samples: int, chunk_id: int) -> list[int]:
        if num_chunks == 0:
            raise ValueError("Number of chunks must be greater than 0.")
        if chunk_id >= num_chunks:
            raise ValueError("Chunk ID must be less than the number of chunks.")

        # get the maximum chunk size given the number of samples and number of chunks
        chunk_size_complete = math.ceil(num_samples / num_chunks)

        # the number of complete chunks, i.e., the chunks having the maximum chunk size
        num_complete_chunks = num_samples % num_chunks
        if num_complete_chunks == 0:
            num_complete_chunks = num_chunks

        # Calculate the start and end index of the chunk
        # The first num_complete_chunks chunks have the maximum chunk size and the
        # remaining ones have chunk_size_complete - 1
        # If the chunk_id is larger than num_complete_chunks, we need calculate the starting position of the chunk
        # by adding chunk_id many offsets of size chunk_size_complete and (chunk_id - num_complete_chunks) many
        # offsets of size chunk_size_complete - 1
        start = chunk_size_complete * min(num_complete_chunks, chunk_id) + max((chunk_id - num_complete_chunks), 0) * (
            chunk_size_complete - 1
        )

        if chunk_id < num_complete_chunks:
            end = start + chunk_size_complete
        else:
            end = start + chunk_size_complete - 1

        return [start, end]

    @staticmethod
    def get_tokenized_file_chunk(dataset: PackedMemMapDatasetBase, num_chunks: int, chunk_id: int) -> list[np.ndarray]:
        chunk_range = Chunking._get_chunk_range(num_chunks=num_chunks, num_samples=len(dataset), chunk_id=chunk_id)
        if chunk_range[0] == chunk_range[1]:
            return []
        chunk = dataset[chunk_range[0] : chunk_range[1]][dataset.sample_key]
        return chunk

    @staticmethod
    def get_jsonl_file_chunk(dataset: list[Any], num_chunks: int, chunk_id: int) -> list[Any]:
        chunk_range = Chunking._get_chunk_range(num_chunks=num_chunks, num_samples=len(dataset), chunk_id=chunk_id)
        chunk = dataset[chunk_range[0] : chunk_range[1]]
        return chunk

    @staticmethod
    def shuffle_file_chunks_in_place(file_chunks: list[Any], seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)  # Create a local random generator
        rng.shuffle(file_chunks)  # Shuffle using the local generator
