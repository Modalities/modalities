import multiprocessing
import pickle
import random
from functools import partial
from multiprocessing import Process, Queue
from pathlib import Path

from modalities.dataloader.create_packed_data import EmbeddedStreamData


def _process_batch_with_embedded_stream(
    batch: list[tuple[int, int]], data: EmbeddedStreamData
) -> tuple[bytes, list[int]]:
    """
    Process a batch of index entries to extract documents and create a new index.

    Args:
        batch (list[tuple[int, int]]): List of index entries [(start, length), ...].
        data (EmbeddedStreamData): Instance of EmbeddedStreamData for accessing the data segment.

    Returns:
        tuple[bytes, list[int]]: A tuple containing the processed data (bytes) and the list of document lengths.
    """
    processed_data = []
    document_lengths = []

    for start, length in batch:
        # Use memmap to read the document from the data segment
        document = data._data[start : start + length]
        processed_data.append(document.tobytes())  # Convert from memmap slice to bytes

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
    """Shuffle data and index segments using EmbeddedStreamData for efficient data access.
    Shuffled data is written to a new file with the postfix "_shuffled".

    Args:
        input_data_path (Path): Path to the tokenized data (.pbin).
        batch_size (int): Number of documents to process per batch.

    Returns:
        None
    """
    # Step 1: Initialize EmbeddedStreamData
    data = EmbeddedStreamData(input_data_path)

    # Step 2: Shuffle the index
    shuffled_index = data._index_base[:]
    random.shuffle(shuffled_index)

    # Step 3: Divide the shuffled index into batches
    batches = [shuffled_index[i : i + batch_size] for i in range(0, len(shuffled_index), batch_size)]

    # Step 4: Prepare for multiprocessing and writing
    queue = Queue()

    header_data = data.data_len.to_bytes(
        EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"
    ) + data.token_size_in_bytes.to_bytes(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little")

    # Get the stem (file name without the extension) and the suffix (extension)
    stem = input_data_path.stem
    suffix = input_data_path.suffix

    # Create the output file path with the postfix "_shuffled"
    output_data_path = input_data_path.with_name(f"{stem}_shuffled{suffix}")

    writer = Process(target=_writer_process, args=(output_data_path, queue, header_data))
    writer.start()

    # Step 5: Use multiprocessing to process batches
    with multiprocessing.Pool() as pool:
        for result in pool.imap(partial(_process_batch_with_embedded_stream, data=data), batches):
            queue.put(result)

    # Step 6: Signal the writer process to finish
    queue.put(None)
    writer.join()


# Example usage
if __name__ == "__main__":
    # Example input file containing header, data, and index segments
    data_path = Path("combined_file.bin")

    # Example output file for shuffled data
    output_path = Path("shuffled_combined_file.bin")

    # Batch size for multiprocessing
    batch_size = 100  # Adjust based on the number of documents and system resources

    # Call the shuffle function with EmbeddedStreamData
    shuffle_tokenized_data(data_path, output_path, batch_size)
