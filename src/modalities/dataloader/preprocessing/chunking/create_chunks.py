import math

import numpy as np

from modalities.dataloader.dataset import PackedMemMapDatasetBase


class Chunking:
    @staticmethod
    def get_chunk_range_(num_chunks: int, num_samples: int, chunk_id: int) -> list[int]:
        # get the maximum chunk size given the number of samples and number of chunks
        chunk_size_complete = math.ceil(num_samples / num_chunks)

        # the number of complete chunks, i.e., the chunks having the maximum chunk size
        num_complete_chunks = num_samples % num_chunks
        if num_complete_chunks == 0:
            num_complete_chunks = num_chunks

        start = chunk_size_complete * min(num_complete_chunks, chunk_id) + max((chunk_id - num_complete_chunks), 0) * (
            chunk_size_complete - 1
        )

        if chunk_id < num_complete_chunks:
            end = start + chunk_size_complete
        else:
            end = start + chunk_size_complete - 1

        return [start, end]

    @staticmethod
    def get_file_chunk(dataset: PackedMemMapDatasetBase, num_chunks: int, chunk_id: int) -> list[np.ndarray]:
        chunk_range = Chunking.get_chunk_range_(num_chunks=num_chunks, num_samples=len(dataset), chunk_id=chunk_id)
        chunk = dataset[chunk_range[0] : chunk_range[1]][dataset.sample_key]
        return chunk

    @staticmethod
    def shuffle_file_chunks(file_chunks: list[np.ndarray]) -> list[np.ndarray]:
        np.random.shuffle(file_chunks)
        return file_chunks
