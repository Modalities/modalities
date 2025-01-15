import multiprocessing
import pickle
import random
from functools import partial
from multiprocessing import Process, Queue
from pathlib import Path

from modalities.dataloader.create_packed_data import EmbeddedStreamData


def _process_batch(batch: list[tuple[int, int]], data: bytes) -> tuple[bytes, list[int]]:
    """
    Process a batch of index entries to extract documents and create a new index.

    Args:
        batch (list[tuple[int, int]]): List of index entries [(start, length), ...].
        data (bytes): Byte stream of the entire data loaded in memory.

    Returns:
        tuple[bytes, list[int]]: A tuple containing the processed data (bytes) and the list of document lengths.
    """
    processed_data = []
    document_lengths = []

    for start, length in batch:
        # Access the data slice directly from the in-memory bytes
        document = data[start : start + length]
        processed_data.append(document)  # Already bytes

        # Record the length of the document
        document_lengths.append(length)

    return b"".join(processed_data), document_lengths


def _writer_process(output_path: Path, queue: Queue, header_data: bytes) -> None:
    """Process to write processed data and index to the output file incrementally.

    Args:
        output_path (Path): Path to the output file.
        queue (Queue): Queue containing processed data and document lengths.
        header_data (bytes): Header data to write initially.

    Returns:
        None
    """
    with output_path.open("wb") as f:
        # Write the header data
        f.write(header_data)
        current_position = 0
        final_index = []

        while True:
            item = queue.get()
            # Sentinel value to indicate completion
            if item is None:
                break

            data_segment, lengths = item
            f.write(data_segment)
            final_index.extend((current_position, length) for length in lengths)
            current_position += len(data_segment)

        # Write the final index to the file
        f.write(pickle.dumps(final_index))


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

    # Step 4: Prepare for multiprocessing and writing
    queue = Queue()

    # Create the output file path with the postfix "_shuffled"
    stem = input_data_path.stem
    suffix = input_data_path.suffix
    output_data_path = input_data_path.with_name(f"{stem}_shuffled{suffix}")

    # Prepare the header data
    header_data = data_section_length_in_bytes + token_size_as_bytes

    writer = Process(target=_writer_process, args=(output_data_path, queue, header_data))
    writer.start()

    # Step 5: Use multiprocessing to process batches
    with multiprocessing.Pool() as pool:
        for result in pool.imap(partial(_process_batch, data=data), batches):
            queue.put(result)

    # Step 6: Signal the writer process to finish
    queue.put(None)
    writer.join()
