import pickle
from pathlib import Path
from random import Random
from typing import Any, MutableSequence, Optional

from modalities.dataloader.create_packed_data import EmbeddedStreamData


class DataShuffler:
    @staticmethod
    def _shuffle_mutable_sequence_in_place(mutable_sequence: MutableSequence[Any], seed=None) -> None:
        """Shuffle a mutable sequence in-place."""
        rng = Random(seed)
        rng.shuffle(mutable_sequence)

    @staticmethod
    def _process_batch(
        batch: list[tuple[int, int]], data: bytes, start_position: int
    ) -> tuple[bytes, list[tuple[int, int]]]:
        """Process a batch of index entries to extract documents and create a new index.

        Args:
            batch (list[tuple[int, int]]): List of index entries [(start, length), ...].
            data (bytes): Byte stream of the entire data loaded in memory.
            start_position (int): The starting position for this batch in the byte stream.

        Returns:
            tuple[bytes, list[tuple[int, int]]]: A tuple containing the processed data (bytes)
            and the new index [(position, length), ...].
        """
        processed_data = []
        new_index = []

        current_position = start_position

        for start, length in batch:
            # Access the data slice directly from the in-memory bytes
            document = data[start : start + length]
            processed_data.append(document)  # Already bytes

            # Record the current position and length in the new index
            new_index.append((current_position, length))
            current_position += length

        return b"".join(processed_data), new_index

    @staticmethod
    def shuffle_tokenized_data(
        input_data_path: Path, output_data_path: Path, batch_size: int, seed: Optional[int] = None
    ) -> None:
        """Shuffles a tokenized file (.pbin).
        Shuffled data is written to the specified output file.

        Note that the tokenized data is fully materialized in-memory.


        Args:
            input_data_path (Path): Path to the tokenized data (.pbin).
            output_data_path (Path): Path to write the shuffled tokenized data.
            batch_size (int): Number of documents to process per batch.
            seed (Optional[int], optional): Seed for the random number generator. Defaults to None.

        Returns:
            None
        """
        # Step 1: Load the entire data into memory
        with input_data_path.open("rb") as f:
            # Read the header
            data_section_length_in_bytes = f.read(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES)
            data_len = int.from_bytes(data_section_length_in_bytes, byteorder="little")

            token_size_as_bytes = f.read(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES)

            # Load the data
            data = f.read(data_len)

            # Load the index
            pkl_encoded_index = f.read()
            index_base = pickle.loads(pkl_encoded_index)

        # Step 2: Shuffle the index
        DataShuffler._shuffle_mutable_sequence_in_place(mutable_sequence=index_base, seed=seed)

        # Step 3: Divide the shuffled index into batches
        batches: list[list[tuple[int, int]]] = [
            index_base[i : i + batch_size] for i in range(0, len(index_base), batch_size)
        ]

        header_data = data_section_length_in_bytes + token_size_as_bytes

        output_data_path.parent.mkdir(parents=True, exist_ok=True)

        with output_data_path.open("wb") as f:
            # Write the header data
            f.write(header_data)
            current_position = 0
            final_index = []

            # Process and write each batch sequentially
            for batch in batches:
                data_segment, new_index = DataShuffler._process_batch(batch, data, current_position)
                f.write(data_segment)
                final_index.extend(new_index)
                current_position += len(data_segment)

            # Write the final index to the file
            f.write(pickle.dumps(final_index))

    @staticmethod
    def shuffle_jsonl_data(input_data_path: Path, output_data_path: Path, seed: Optional[int] = None):
        # read jsonl and store in RAM
        with input_data_path.open("r") as f:
            data = f.readlines()

        DataShuffler._shuffle_mutable_sequence_in_place(data, seed)
        with output_data_path.open("w") as f:
            f.writelines(data)
