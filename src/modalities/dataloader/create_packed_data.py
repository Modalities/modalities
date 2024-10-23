import logging
import math
import multiprocessing
import os
import pickle
import warnings
from io import BufferedWriter
from pathlib import Path
from typing import Callable, Iterator, Optional

import jq
import numpy as np
from pydantic import FilePath
from tqdm import tqdm

from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)


class EmptySampleError(RuntimeError):
    pass


class PackedDataGenerator:
    """Reads in a JSONL file and the corresponding index file and packs the dataset for LLM training."""

    def __init__(
        self,
        src_path: FilePath,
        tokenizer: TokenizerWrapper,
        eod_token: str,
        number_of_processes: int,
        jq_pattern: str,
        processing_batch_size: int,
        raw_samples_queue_size: int,
        processed_samples_queue_size: int,
        index_path: Optional[FilePath] = None,
    ):
        """
        Initializes a PackedDataGenerator object.

        Args:
            src_path (FilePath): Path to a JSONL file, which holds text data.
            tokenizer (TokenizerWrapper): PretrainedTokenizer object used to tokenize the provided data in `src_path`.
            eod_token (str): End-of-document token.
            number_of_processes (int): Number of processes used for parallel processing.
            jq_pattern (str): jq-pattern applied on every jsonl-entry. Results are afterwards tokenized and packed.
            processing_batch_size (int): Size of the batches that the workers process.
            raw_samples_queue_size (int): Maximum size of the raw samples queue.
            processed_samples_queue_size (int): Maximum size of the processed samples queue.
            index_path (Optional[FilePath], optional): Path to an index file,
                which indicates the start character position
                and length of samples given in `src_path`. If not defined, an index file next to `src_path` is picked,
                by replacing its suffix with ".idx". Defaults to None.

        Returns:
            None
        """
        self.src_path = src_path
        self.tokenizer = tokenizer
        self.eod_token = eod_token
        self._token_size_in_bytes = self._get_required_num_of_bytes_to_repr(self.tokenizer.vocab_size)
        encoded_eod_token = self.tokenizer.get_token_id(self.eod_token)
        self._encoded_eos_token_as_bytes = self._encoded_token_to_bytes(encoded_eod_token)
        self.jq_filter = jq.compile(jq_pattern)
        self._number_of_processes = number_of_processes
        self._reader = LargeFileLinesReader(src_path, index_path=index_path)
        self._total_num_of_tokens = 0
        self._raw_samples_queue = multiprocessing.Queue(maxsize=raw_samples_queue_size)
        self.processed_samples_queue = multiprocessing.Queue(maxsize=processed_samples_queue_size)
        self._exception_buffer = []
        self.processing_batch_size = processing_batch_size

    @staticmethod
    def _get_required_num_of_bytes_to_repr(int_to_get_repr: int) -> int:
        """
        Calculates the required number of bytes to represent an integer.

        Args:
            int_to_get_repr (int): The integer to get the representation for.

        Returns:
            int: The number of bytes required to represent the integer.
        """
        return math.ceil(math.log(math.log2(int_to_get_repr), 8))

    def _encoded_token_to_bytes(self, encoded_token: int) -> bytes:
        """
        Converts an encoded token to its byte representaion.

        Args:
            encoded_token (int): The encoded token to be converted.

        Returns:
            bytes: The byte representation of the token.

        """
        return encoded_token.to_bytes(self._token_size_in_bytes, byteorder="little", signed=False)

    def _default_destination_path(self, destination_path: Optional[Path] = None) -> Path:
        """
        Returns the default destination path for the packed data.

        Args:
            destination_path (Path, optional): The specific destination path. Defaults to None.

        Returns:
            Path: The default destination path for the packed data.
        """
        if destination_path is None:
            default_destination_path = Path(self.src_path.parent, f"{self.src_path.stem}.pbin")
            print(
                f"No specific Destination Path provided. "
                f"Pointing to destination next to input data at: {default_destination_path}"
            )
            return default_destination_path
        return Path(destination_path)

    def run(self, dst_path: Optional[Path] = None):
        """
        Packs data and saves it to (default) dst_path.

        Args:
            dst_path (Optional[Path]): The destination path to save the packed data.
            If not provided, a default destination path will be used.

        Raises:
            ValueError: If the file already exists at the destination path.
            Exception: If an exception occurs during the data packing process.

        Returns:
            None
        """
        assert self._total_num_of_tokens == 0, f"This {self.__name__} was already used and is exhausted. Use another!"
        dst_path = self._default_destination_path(destination_path=dst_path)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            raise ValueError(f"file already exists at destination path '{dst_path}'.")

        self._exception_buffer = []
        try:
            # not setting this can cause deadlocks when using hf's "FastTokenizers". See also:
            # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/67254879#67254879
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self._launch_parallelized_workers(dst_path)
        finally:
            os.unsetenv("TOKENIZERS_PARALLELISM")

        if self._exception_buffer:
            raise self._exception_buffer[0]

    def _launch_parallelized_workers(self, dst_path: Path):
        # Launches workers in parallel for reading, writing, and processing data.
        # The data is stored in the provided destination path.

        reader = multiprocessing.Process(target=self._reader_thread())
        reader.start()

        writer = multiprocessing.Process(target=self._writer_thread(dst_path))
        writer.start()
        processor_threads = [
            multiprocessing.Process(target=self._process_thread, args=(i,)) for i in range(self._number_of_processes)
        ]
        for p in processor_threads:
            p.start()
        for p in processor_threads:
            p.join()
        self._stop_processing()
        writer.join()

    def _stop_processing(self):
        # Stops the processing of samples by putting None in the processed_samples_queue.
        self.processed_samples_queue.put(None)

    def _generator_for_tokens_to_get_written(self):
        # Generator function that yields batches of processed samples.

        while True:
            if self._check_for_parallel_errors():
                return
            batch = self.processed_samples_queue.get()
            if batch is None:
                break
            yield batch

    def _check_for_parallel_errors(self) -> bool:
        # Checks if there are any errors in the exception buffer.
        return bool(self._exception_buffer)

    def _writer_thread(self, dst_path: Path) -> Callable:
        # Returns a callable writer function that writes a batch
        # received from the processed_samples_queue to the destination file.

        def writer():
            # writes a batch received from the processed_samples_queue to the destination file
            def _write_batch(
                batch: list[tuple[int, bytes]], prev_line_id: int, curr_offset: int, index_list: list, f: BufferedWriter
            ) -> tuple[int, int]:
                # write the tokens for each document
                for line_id, tokens_as_bytes in batch:
                    if prev_line_id + 1 != line_id:
                        raise ValueError(
                            f"Line IDs are not consecutive. Expected {prev_line_id + 1}, but got {line_id}"
                        )
                    f.write(tokens_as_bytes)
                    segment_length = len(tokens_as_bytes)
                    index_list.append((curr_offset, segment_length))
                    curr_offset += segment_length
                    prev_line_id = line_id
                return prev_line_id, curr_offset

            index_list = []
            with dst_path.open("wb") as f:
                # allocate first self.header_size_in_bytes bytes for header (encodes length of data section)
                # not possible to prepend header after determining size of data section
                f.write((0).to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"))
                f.write(
                    self._token_size_in_bytes.to_bytes(
                        EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little"
                    )
                )
                # The offset only applies to the data section, not the header
                # When we load the file, we add the header size to the offset
                curr_offset = 0

                # write data section (tokens)
                pbar = tqdm(total=len(self._reader), desc="Processed batches")
                prev_line_id = -1
                batch_dict = {}
                for batch in self._generator_for_tokens_to_get_written():
                    line_id = batch[0][0]
                    batch_dict[line_id] = batch

                    while prev_line_id + 1 in batch_dict:
                        batch = batch_dict.pop(prev_line_id + 1)
                        prev_line_id, curr_offset = _write_batch(batch, prev_line_id, curr_offset, index_list, f)
                        pbar.update(len(batch))
                # write index
                f.write(pickle.dumps(index_list))

            self._update_data_length_in_pre_allocated_header(dst_path, index_list)

        return writer

    def _reader_thread(self) -> Callable:
        # returns a reader function that reads lines from the reader and puts them into a queue.
        def reader():
            batch = []
            for line_id, line in tqdm(enumerate(self._reader), desc="Reading jsonl", disable=True):
                # line = self._reader[line_id]
                batch.append((line_id, line))
                if len(batch) % self.processing_batch_size == 0:
                    self._raw_samples_queue.put(batch)
                    batch = []

            # add the remaining samples
            if len(batch) > 0:
                self._raw_samples_queue.put(batch)

            for _ in range(self._number_of_processes):
                self._raw_samples_queue.put(None)

        return reader

    def _process_thread(self, process_id: int):
        # Process the lines in a batch and put the processed samples into the processed_samples_queue.
        if self._check_for_parallel_errors():
            return

        while True:
            if self._check_for_parallel_errors():
                return
            batch = self._raw_samples_queue.get()
            if batch is None:
                break

            try:
                batch_processed = []
                for line_id, line in batch:
                    processed_line = self._process_line(line, process_id)
                    batch_processed.append((line_id, processed_line))
                self.processed_samples_queue.put(batch_processed)
            except EmptySampleError:
                warnings.warn(
                    f"Encountered empty sample in line {line_id} of file {self.src_path} within process {process_id}"
                )
            except Exception as exception:
                warnings.warn(
                    f"Could not process line of number {line_id} within process {process_id}. "
                    f"Raised the following error: {exception=}"
                )

    def _update_data_length_in_pre_allocated_header(self, dst_path: Path, index_list: list[tuple[int, int]]):
        # Update the length of the data section in the pre-allocated header of the destination file.
        # The data segment length is sum of the starting position and the length of the last document.
        length_of_byte_encoded_data_section = index_list[-1][0] + index_list[-1][1]
        data_section_length_in_bytes = length_of_byte_encoded_data_section.to_bytes(
            EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"
        )
        with dst_path.open("rb+") as fout:
            fout.seek(0)
            fout.write(data_section_length_in_bytes)

    def _process_line(self, line: str, process_id: int) -> bytes:
        # extracts the text via the jq_filter and applies tokenization to the extract text
        jq_retrieved_text = self.jq_filter.input_text(line).first()
        if jq_retrieved_text is None:
            raise ValueError(f"jq was not able to find anything using the expression: {self.jq_filter}")
        tokens = self.tokenizer.tokenize(jq_retrieved_text)
        if len(tokens) == 0:
            raise EmptySampleError("Received empty sample...")
        return b"".join(map(self._encoded_token_to_bytes, tokens)) + self._encoded_eos_token_as_bytes


class EmbeddedStreamData:
    # amount of bytes to represent number of all tokens in dataset.
    # If the amount exceeds 2^(8*`header_size_in_bytes`), this requires adaptation.
    # Decided to keep this constant, since a size of 8 bytes requires more data than the internet currently provides
    DATA_SECTION_LENGTH_IN_BYTES = 8
    TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES = 4
    HEADER_SIZE_IN_BYTES = DATA_SECTION_LENGTH_IN_BYTES + TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES

    def __init__(self, data_path: Path):
        """
        Initializes an EmbeddedStreamData object.

        Args:
            data_path (Path): The path to the packed data file.

        Raises:
            FileNotFoundError: If the packed data file is not found at the specified path.

        """
        self._data_path = data_path
        if not self._data_path.is_file():
            raise FileNotFoundError(
                f"Packed Data was not found at {self._data_path.absolute()}."
                f"Create on in advance by using `modalities data pack_encoded_data`."
            )

        with self._data_path.open("rb") as f:
            # get number of bytes in data section
            data_section_length_in_bytes = f.read(self.DATA_SECTION_LENGTH_IN_BYTES)
            self.data_len = int.from_bytes(data_section_length_in_bytes, byteorder="little")

            # get number of bytes for encoding a single token
            f.seek(self.DATA_SECTION_LENGTH_IN_BYTES)
            token_size_as_bytes = f.read(self.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES)
            self.token_size_in_bytes = int.from_bytes(token_size_as_bytes, byteorder="little", signed=False)

            # get index
            f.seek(self.HEADER_SIZE_IN_BYTES + self.data_len)
            pkl_encoded_index = f.read()
            # contains the start offset and length of each segment
            # as byte positions in the data section
            self.index_base: list[tuple[int, int]] = pickle.loads(pkl_encoded_index)

            # initialize memmapped data section
            self.data = np.memmap(self._data_path, mode="r", offset=self.HEADER_SIZE_IN_BYTES, shape=(self.data_len,))


def join_embedded_stream_data(stream_data: list[EmbeddedStreamData], target_file: Path, chunk_size: int = 2048):
    """
    Joins the embedded stream data into a single file.

    Args:
        stream_data (list[EmbeddedStreamData]): A list of EmbeddedStreamData objects representing the stream data.
        target_file (Path): The target file to write the joined data to.
        chunk_size (int, optional): The size of each data chunk. Defaults to 2048.

    Raises:
        FileExistsError: If the target file already exists.

    Returns:
        None
    """
    if target_file.exists():
        raise FileExistsError(f'Target File at "{target_file}" exists!')
    data_len = sum(d.data_len for d in stream_data)
    assert len({d.token_size_in_bytes for d in stream_data}) == 1, (
        "Found different token representation sizes. This could indicate the usage of different tokenizers. "
        "Not supported!"
    )
    token_size_in_bytes = stream_data[0].token_size_in_bytes

    num_data_chunks = sum(math.ceil(d.data_len / chunk_size) for d in stream_data)
    data_stream_generator = (d.data[i : i + chunk_size] for d in stream_data for i in range(0, d.data_len, chunk_size))

    num_entries = sum(len(d.index_base) for d in stream_data)

    def index_stream_generator() -> Iterator[tuple[int, int]]:
        # generates a stream of index offsets and segment lengths.
        curr_offset = 0
        for embedded_stream_data in stream_data:
            for entry_offset, segment_length in embedded_stream_data.index_base:
                yield entry_offset + curr_offset, segment_length
            curr_offset += embedded_stream_data.data_len
            curr_offset -= embedded_stream_data.HEADER_SIZE_IN_BYTES

    with target_file.open("wb") as fout:
        fout.write(data_len.to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"))
        fout.write(
            token_size_in_bytes.to_bytes(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little")
        )
        for data_chunk in tqdm(data_stream_generator, total=num_data_chunks, desc="Writing Data Chunks..."):
            fout.write(data_chunk)

        joint_index = [entry for entry in tqdm(index_stream_generator(), total=num_entries, desc="Concatenating Index")]
        pickled_index = pickle.dumps(joint_index)
        pickled_index_as_chunks = (pickled_index[i : i + chunk_size] for i in range(0, len(pickled_index), chunk_size))
        num_index_chunks = math.ceil(len(pickled_index) / chunk_size)
        for index_chunk in tqdm(pickled_index_as_chunks, total=num_index_chunks, desc="Writing Index Chunks..."):
            fout.write(index_chunk)
