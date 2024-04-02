import logging
import math
import multiprocessing
import os
import pickle
import warnings
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import jq
import numpy as np
from pydantic import FilePath
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader

logger = logging.getLogger(__name__)


class EmptySampleError(RuntimeError):
    pass


class PackedDataGenerator:
    def __init__(
        self,
        src_path: FilePath,
        tokenizer: PreTrainedTokenizer,
        number_of_processes: int,
        index_path: FilePath,
        jq_pattern: str,
    ):
        """
        Reads in a jsonl file and the corresponding index file and packs dataset file for LLM training.
        :param src_path: Path to a jsonl file, which holds text data
        :param index_path: Path to an index file, which indicates the start character position
                           and length of samples given in `src_path`.
                           If not defined, an index file next to `src_path` is picked,
                           by replacing its suffix with ".idx".
        :param tokenizer: PretrainedTokenizer object, which is used to pre-tokenize the provided data in `src_path`.
                          Tokenization is necessary to work on final lengths of token sequences.
        :param jq_pattern: jq-pattern applied on every jsonl-entry. Results are afterwards tokenized and packed
        """
        self.src_path = src_path
        self.tokenizer = tokenizer
        self._token_size_in_bytes = self._get_required_num_of_bytes_to_repr(self.tokenizer.vocab_size)
        encoded_eos_token = self.tokenizer(self.tokenizer.eos_token)["input_ids"][0]
        self._encoded_eos_token_as_bytes = self._encoded_token_to_bytes(encoded_eos_token)
        self.jq_filter = jq.compile(jq_pattern)
        self._number_of_processes = number_of_processes
        self._reader = LargeFileLinesReader(src_path, index_path=index_path)
        self._total_num_of_tokens = 0
        self._tokens_write_queue = multiprocessing.Queue()
        self._exception_buffer = []

    @staticmethod
    def _get_required_num_of_bytes_to_repr(int_to_get_repr: int) -> int:
        return math.ceil(math.log(math.log2(int_to_get_repr), 8))

    def _encoded_token_to_bytes(self, encoded_token: int) -> bytes:
        return encoded_token.to_bytes(self._token_size_in_bytes, byteorder="big", signed=False)

    def _default_destination_path(self, destination_path: Path = None) -> Path:
        if destination_path is None:
            default_destination_path = Path(self.src_path.parent, f"{self.src_path.stem}.pbin")
            print(
                f"No specific Destination Path provided. "
                f"Pointing to destination next to input data at: {default_destination_path}"
            )
            return default_destination_path
        return Path(destination_path)

    def run(self, dst_path: Path = None):
        assert self._total_num_of_tokens == 0, f"This {self.__name__} was already used and is exhausted. Use another!"
        dst_path = self._default_destination_path(destination_path=dst_path)

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
        self._tokens_write_queue.put(None)

    def _generator_for_tokens_to_get_written(self):
        while True:
            if self._check_for_parallel_errors():
                return
            tokens = self._tokens_write_queue.get()
            if tokens is None:
                break
            yield tokens

    def _check_for_parallel_errors(self) -> bool:
        return bool(self._exception_buffer)

    def _writer_thread(self, dst_path: Path) -> Callable:
        def writer():
            index_list = []
            with dst_path.open("wb") as f:
                # allocate first self.header_size_in_bytes bytes for header (encodes length of data section)
                # not possible to prepend header after determining size of data section
                f.write((0).to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="big"))
                f.write(
                    self._token_size_in_bytes.to_bytes(
                        EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="big"
                    )
                )
                curr_offset = EmbeddedStreamData.HEADER_SIZE_IN_BYTES

                # write data section (tokens)
                for tokens_as_bytes in tqdm(
                    self._generator_for_tokens_to_get_written(), desc="Processed Samples", total=len(self._reader)
                ):
                    f.write(tokens_as_bytes)
                    segment_length = len(tokens_as_bytes)
                    index_list.append((curr_offset, segment_length))
                    curr_offset += segment_length

                # write index
                f.write(pickle.dumps(index_list))

            self._update_data_length_in_pre_allocated_header(dst_path, index_list)

        return writer

    def _process_thread(self, process_id: int):
        if self._check_for_parallel_errors():
            return
        for idx in range(process_id, len(self._reader), self._number_of_processes):
            line = self._reader[idx]
            try:
                self._tokens_write_queue.put(self._process_line(line))
            except EmptySampleError:
                warnings.warn(f"Encountered empty sample in line {idx} of file {self.src_path}")
            except Exception as exception:
                warnings.warn(f"could not process line of number {idx}. Raised the following error: {exception=}")

    def _update_data_length_in_pre_allocated_header(self, dst_path: Path, index_list: List[Tuple[int, int]]):
        start_of_index_in_bytes = index_list[-1][0] + index_list[-1][1]
        length_of_byte_encoded_data_section = start_of_index_in_bytes - EmbeddedStreamData.HEADER_SIZE_IN_BYTES
        data_section_length_in_bytes = length_of_byte_encoded_data_section.to_bytes(
            EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="big"
        )
        with dst_path.open("rb+") as fout:
            fout.seek(0)
            fout.write(data_section_length_in_bytes)

    def _process_line(self, line: str) -> bytes:
        jq_retrieved_text = self.jq_filter.input_text(line).first()
        if jq_retrieved_text is None:
            raise ValueError(f"jq was not able to find anything using the expression: {self.jq_filter}")
        tokens = self.tokenizer(jq_retrieved_text)["input_ids"]
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
        self._data_path = data_path
        if not self._data_path.is_file():
            raise FileNotFoundError(
                f"Packed Data was not found at {self._data_path}."
                f"Create on in advance by using `modalities data pack_encoded_data`."
            )

        with self._data_path.open("rb") as f:
            # get number of bytes in data section
            data_section_length_in_bytes = f.read(self.DATA_SECTION_LENGTH_IN_BYTES)
            self.data_len = int.from_bytes(data_section_length_in_bytes, byteorder="big")

            # get number of bytes for encoding a single token
            f.seek(self.DATA_SECTION_LENGTH_IN_BYTES)
            token_size_as_bytes = f.read(self.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES)
            self.token_size_in_bytes = int.from_bytes(token_size_as_bytes, byteorder="big", signed=False)

            # get index
            f.seek(self.HEADER_SIZE_IN_BYTES + self.data_len)
            pkl_encoded_index = f.read()
            self.index_base = pickle.loads(pkl_encoded_index)

            # initialize memmapped data section
            self.data = np.memmap(self._data_path, mode="r", offset=self.HEADER_SIZE_IN_BYTES, shape=(self.data_len,))


def join_embedded_stream_data(stream_data: List[EmbeddedStreamData], target_file: Path, chunk_size: int = 2048):
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

    def index_stream_generator() -> Iterator[Tuple[int, int]]:
        curr_offset = 0
        for embedded_stream_data in stream_data:
            for entry_offset, segment_length in embedded_stream_data.index_base:
                yield entry_offset + curr_offset, segment_length
            curr_offset += embedded_stream_data.data_len
            curr_offset -= embedded_stream_data.HEADER_SIZE_IN_BYTES

    with target_file.open("wb") as fout:
        fout.write(data_len.to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="big"))
        fout.write(
            token_size_in_bytes.to_bytes(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="big")
        )
        for data_chunk in tqdm(data_stream_generator, total=num_data_chunks, desc="Writing Data Chunks..."):
            fout.write(data_chunk)

        joint_index = [entry for entry in tqdm(index_stream_generator(), total=num_entries, desc="Concatenating Index")]
        pickled_index = pickle.dumps(joint_index)
        pickled_index_as_chunks = (pickled_index[i : i + chunk_size] for i in range(0, len(pickled_index), chunk_size))
        num_index_chunks = math.ceil(len(pickled_index) / chunk_size)
        for index_chunk in tqdm(pickled_index_as_chunks, total=num_index_chunks, desc="Writing Index Chunks..."):
            fout.write(index_chunk)
