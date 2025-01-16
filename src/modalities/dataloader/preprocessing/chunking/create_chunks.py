import math


class Chunking:
    @staticmethod
    def get_chunk_range(num_chunks: int, num_samples: int, chunk_id: int) -> list[int]:
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
