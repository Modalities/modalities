import pickle
import random
from pathlib import Path

from modalities.dataloader.create_packed_data import EmbeddedStreamData


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


def shuffle_tokenized_data(input_data_path: Path, batch_size: int) -> None:
    """Shuffle data and index segments loaded fully into memory.
    Shuffled data is written to a new file with the postfix "_shuffled".

    Args:
        input_data_path (Path): Path to the tokenized data (.pbin).
        batch_size (int): Number of documents to process per batch.

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
    random.shuffle(index_base)

    # Step 3: Divide the shuffled index into batches
    batches = [index_base[i : i + batch_size] for i in range(0, len(index_base), batch_size)]

    # Step 4: Prepare the output file
    stem = input_data_path.stem
    suffix = input_data_path.suffix
    output_data_path = input_data_path.with_name(f"{stem}_shuffled{suffix}")

    header_data = data_section_length_in_bytes + token_size_as_bytes

    with output_data_path.open("wb") as f:
        # Write the header data
        f.write(header_data)
        current_position = 0
        final_index = []

        # Process and write each batch sequentially
        for batch in batches:
            data_segment, new_index = _process_batch(batch, data, current_position)
            f.write(data_segment)
            final_index.extend(new_index)
            current_position += len(data_segment)

        # Write the final index to the file
        f.write(pickle.dumps(final_index))
