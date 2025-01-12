import math
import multiprocessing as mp
import os
import pickle
import time
from io import BufferedWriter
from pathlib import Path
from typing import Type

import jq
from data_quality_ablations.utils.logging import get_logger
from pydantic import BaseModel

from modalities.config.instantiation_models import TokenizationInstantiationModel
from modalities.dataloader.preprocessing.queued_processing.processing_strategy_if import ProcessingStrategyIF
from modalities.dataloader.preprocessing.queued_processing.queue_items import ProgressMessage, ReadingJob
from modalities.dataloader.preprocessing.tokenization.embedded_stream_data import EmbeddedStreamData
from modalities.dataloader.preprocessing.tokenization.large_file_lines_reader import (
    BaseReaderIF,
    LargeFileLinesReaderFactory,
    LargeFileLinesReaderTypes,
)
from modalities.dataloader.preprocessing.tokenization.queue_items import Sample
from modalities.dataloader.preprocessing.tokenization.worker_types import WorkerTypes
from modalities.exceptions import EmptySampleError, ProcessingStrategyDoneException
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


def get_required_num_of_bytes_to_repr(int_to_get_repr: int) -> int:
    """
    Calculates the required number of bytes to represent an integer.

    Args:
        int_to_get_repr (int): The integer to get the representation for.

    Returns:
        int: The number of bytes required to represent the integer.
    """
    # we currently only support token sizes of 1, 2 and 4 bytes, as implemented here:
    # https://github.com/Modalities/modalities/blob/fix_char_bytes_indexation_mismatch/src/modalities/dataloader/dataset.py#L202
    num_bytes = math.ceil(math.log2(int_to_get_repr) / 8)
    if num_bytes == 1:
        return 1
    elif num_bytes == 2:
        return 2
    elif num_bytes <= 4:
        return 4
    else:
        raise ValueError("Currently only support token byte sizes of 1, 2, and 4.")


class PopulatingStrategy(ProcessingStrategyIF):
    def __init__(
        self, reader_q_key: str, logging_message_q_key: str, index_start: int, num_samples: int, batch_size: int
    ):
        self._reader_q_key = reader_q_key
        self._logging_message_q_key = logging_message_q_key
        self._batch_size = batch_size
        self._reading_iter = iter(range(index_start, index_start + num_samples, batch_size))

    def __enter__(self):
        return self

    def finalize(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def process(self) -> dict[str, ReadingJob | ProgressMessage]:
        try:
            sample_id = next(self._reading_iter)
        except StopIteration:
            raise ProcessingStrategyDoneException("PopulatingStrategy done.")
        reading_job = ReadingJob(sample_id=sample_id, batch_size=self._batch_size)
        progress_message = ProgressMessage(WorkerTypes.POPULATOR, num_samples=self._batch_size)
        return {self._reader_q_key: reading_job, self._logging_message_q_key: progress_message}


class ReadingStrategy(ProcessingStrategyIF):
    def __init__(
        self, reader_type: Type[BaseReaderIF], reader_args: BaseModel, tokenizer_q_key: str, logging_message_q_key: str
    ):
        self._reader_type = reader_type
        self._reader_args = reader_args
        self._reader = None
        self._tokenizer_q_key = tokenizer_q_key
        self._logging_message_q_key = logging_message_q_key

    def __enter__(self):
        self._reader = self._reader_type(**self._reader_args.model_dump())
        return self

    def finalize(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._reader.close()

    def process(self, item: ReadingJob) -> dict[str, list[Sample] | ProgressMessage]:
        batch: list[Sample] = [self._reader[item.sample_id + i] for i in range(item.batch_size)]
        progress_message = ProgressMessage(WorkerTypes.READER, len(batch))
        return {self._tokenizer_q_key: batch, self._logging_message_q_key: progress_message}


class TokenizingStrategy(ProcessingStrategyIF):
    def __init__(
        self,
        ti_settings: (
            TokenizationInstantiationModel.TokenizerWorkerSettings.TokenizerSettings.TokenizerInstantitionSettings
        ),
        eod_token: str,
        jq_pattern: str,
        writer_q_key: str,
        logging_message_q_key: str,
    ):
        self._tokenizer_instantiation_setings = ti_settings
        self._eod_token = eod_token
        self._jq_filter = jq.compile(jq_pattern)
        self._writer_q_key = writer_q_key
        self._logging_message_q_key = logging_message_q_key

    def __enter__(self):
        registry = Registry(COMPONENTS)
        tokenizer_type: Type[TokenizerWrapper] = registry.get_component(
            component_key=self._tokenizer_instantiation_setings.tokenizer_component_key,
            variant_key=self._tokenizer_instantiation_setings.tokenizer_variant_key,
        )
        self._tokenizer: TokenizerWrapper = tokenizer_type(**self._tokenizer_instantiation_setings.config)

        encoded_eod_token = self._tokenizer.get_token_id(self._eod_token)
        self._token_size_in_bytes = get_required_num_of_bytes_to_repr(self._tokenizer.vocab_size)
        self._encoded_eos_token_as_bytes = TokenizingStrategy._encoded_token_to_bytes(
            token_size_in_bytes=self._token_size_in_bytes, encoded_token=encoded_eod_token
        )
        return self

    def finalize(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def process(self, item: list[Sample]) -> dict[str, list[Sample] | ProgressMessage]:
        batch_processed = []
        for sample in item:
            processed_line = self._process_line(sample.content_raw)
            sample.content_tokenized = processed_line
            sample.token_size_in_bytes = self._token_size_in_bytes
            batch_processed.append(sample)
        progress_message = ProgressMessage(WorkerTypes.TOKENIZER, num_samples=len(batch_processed))
        return {self._writer_q_key: batch_processed, self._logging_message_q_key: progress_message}

    def _process_line(self, line: str) -> bytes:
        # extracts the text via the jq_filter and applies tokenization to the extract text
        jq_retrieved_text = self._jq_filter.input_text(line).first()
        if jq_retrieved_text is None:
            raise ValueError(f"jq was not able extract the text using the expression: {self._jq_filter}")
        tokens = self._tokenizer.tokenize(jq_retrieved_text)
        if len(tokens) == 0:
            raise EmptySampleError("Received empty sample...")
        return (
            b"".join(map(self._encoded_token_to_bytes, [self._token_size_in_bytes] * len(tokens), tokens))
            + self._encoded_eos_token_as_bytes
        )

    @staticmethod
    def _encoded_token_to_bytes(token_size_in_bytes: int, encoded_token: int) -> bytes:
        # Converts an encoded token to its bytes representaion.
        return encoded_token.to_bytes(token_size_in_bytes, byteorder="little", signed=False)


class WritingStrategy(ProcessingStrategyIF):
    def __init__(self, dst_path: Path, index_start: int, logging_message_q_key: str):
        self._dst_path = dst_path
        self._index_start = index_start
        self._logging_message_q_key = logging_message_q_key

        if not self._dst_path.parent.exists():
            self._dst_path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self._dst_fd = self._dst_path.open("wb")
        self.finalized = False
        # allocate first self.header_size_in_bytes bytes for header (encodes length of data section)
        # not possible to prepend header after determining size of data section
        self._dst_fd.write((0).to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"))

        # The offset only applies to the data section, not the header
        # When we load the file, we add the header size to the offset
        self._curr_offset = 0

        self._prev_line_id = self._index_start - 1
        self._batch_dict = {}
        self._index_list = []
        self._has_seen_first_batch = False

        return self

    def finalize(self):
        # check that the index list IS NOT empty and the batch_dict IS empty
        # i.e., all batches have been written to the file
        if len(self._index_list) == 0 or len(self._batch_dict) > 0:
            raise ValueError(
                f"Could not finalize writing strategy. Index list is empty or batch_dict is not empty. "
                f"Index list: {len(self._index_list)}, batch_dict: {self._batch_dict.keys()}"
            )
        else:
            # write index
            self._dst_fd.write(pickle.dumps(self._index_list))
            self._dst_fd.close()
            self._update_data_length_in_pre_allocated_header(self._dst_path, self._index_list)
            self.finalized = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.finalized:
            self._dst_fd.close()
            # if the process was stopped due to a stop event or the index list is empty, we remove the file
            get_logger(name="main").warning(
                f"Removing file {self._dst_path} due to non-finalized pbin file. The pbin file either is not "
                "finalized as WritingStrategy.finalize() was not called or not all samples have been written "
                f"to disc. index_list: {len(self._index_list)}, batch_dict: {self._batch_dict.keys()}"
            )
            os.remove(self._dst_path)

    def process(self, item: list[Sample]) -> dict[str, ProgressMessage]:
        if not self._has_seen_first_batch:
            # write the token size descriptor to the file
            # we receive this information from the tokenizer (based on the tokenizer's vocab size)
            # and is always provided within the Sample object
            self._has_seen_first_batch = True
            self._dst_fd.write(
                item[0].token_size_in_bytes.to_bytes(
                    EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little"
                )
            )

        line_id = item[0].incremental_line_id
        self._batch_dict[line_id] = item

        num_samples_written = 0
        while self._prev_line_id + 1 in self._batch_dict:
            batch = self._batch_dict.pop(self._prev_line_id + 1)
            self._prev_line_id, self._curr_offset = WritingStrategy._write_batch(
                batch, self._prev_line_id, self._curr_offset, self._index_list, self._dst_fd
            )
            num_samples_written += len(batch)
        progress_message = ProgressMessage(WorkerTypes.WRITER, num_samples=num_samples_written)
        return {self._logging_message_q_key: progress_message}

    # writes a batch received from the writer_q to the destination file
    @staticmethod
    def _write_batch(
        batch: list[Sample], prev_line_id: int, curr_offset: int, index_list: list, f: BufferedWriter
    ) -> tuple[int, int]:
        # write the tokens for each document
        for sample in batch:
            if prev_line_id + 1 != sample.incremental_line_id:
                raise ValueError(
                    f"Line IDs are not consecutive. Expected {prev_line_id + 1}, but got {sample.incremental_line_id}"
                )
            f.write(sample.content_tokenized)
            segment_length = len(sample.content_tokenized)
            index_list.append((curr_offset, segment_length))
            curr_offset += segment_length
            prev_line_id = sample.incremental_line_id
        return prev_line_id, curr_offset

    @staticmethod
    def _update_data_length_in_pre_allocated_header(dst_path: Path, index_list: list[tuple[int, int]]):
        # Update the length of the data section in the pre-allocated header of the destination file.
        # The data segment length is sum of the starting position and the length of the last document.
        length_of_byte_encoded_data_section = index_list[-1][0] + index_list[-1][1]
        data_section_length_in_bytes = length_of_byte_encoded_data_section.to_bytes(
            EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"
        )
        with dst_path.open("rb+") as fout:
            fout.seek(0)
            fout.write(data_section_length_in_bytes)


class ProgressLoggingStrategy(ProcessingStrategyIF):
    def __init__(
        self,
        logging_interval: int,
        total_num_samples: int,
        q_dict: dict[str, mp.Queue],
    ):
        self._logging_interval = logging_interval
        self._total_num_samples = total_num_samples
        self._worker_to_pid_to_num_samples: dict[WorkerTypes, dict[int, int]] = {}
        self._worker_type_to_processed_num_samples = {worker_type: 0 for worker_type in WorkerTypes}
        self._q_dict = q_dict

    def __enter__(self):
        self._last_logged = time.time()

    def finalize(self):
        passed_time = time.time() - self._last_logged
        self._log_and_reset(passed_time)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def process(self, item: ProgressMessage) -> dict:
        self._add_progress_message(item)
        passed_time = time.time() - self._last_logged
        if passed_time > self._logging_interval:
            self._log_and_reset(passed_time)
            self._last_logged = time.time()
        return {}

    def _add_progress_message(self, progress_message: ProgressMessage):
        if progress_message.worker_type not in self._worker_to_pid_to_num_samples:
            self._worker_to_pid_to_num_samples[progress_message.worker_type] = {}

        if progress_message.process_id not in self._worker_to_pid_to_num_samples[progress_message.worker_type]:
            self._worker_to_pid_to_num_samples[progress_message.worker_type][progress_message.process_id] = 0

        self._worker_to_pid_to_num_samples[progress_message.worker_type][
            progress_message.process_id
        ] += progress_message.num_samples
        self._worker_type_to_processed_num_samples[progress_message.worker_type] += progress_message.num_samples

    def _log_and_reset(self, passed_time: int):
        logging_message = f"\n==================Progress report (last {passed_time}s) ==================\n"

        logging_message += "Total progress: \n"
        for worker_type, processed_num_samples in self._worker_type_to_processed_num_samples.items():
            m = (
                f"\t{worker_type.name}: {processed_num_samples}/{self._total_num_samples} samples "
                f"({processed_num_samples/self._total_num_samples*100}%)\n"
            )
            logging_message += m

        logging_message += "\n"
        logging_message += "Aggregated Throughput: \n"

        for worker_type, pid_to_num_samples in self._worker_to_pid_to_num_samples.items():
            total_samples = sum(pid_to_num_samples.values())
            logging_message += f"\t{worker_type.name} workers: {total_samples/passed_time} samples/s.\n"
        logging_message += "\n"
        logging_message += "Worker Throughput: \n"
        for worker_type, pid_to_num_samples in self._worker_to_pid_to_num_samples.items():
            logging_message += f"{worker_type.name} workers:\n"
            for pid, num_samples in pid_to_num_samples.items():
                logging_message += f"\t{worker_type.name} {pid}: {num_samples/passed_time} samples/s.\n"
            logging_message += "\n"
        logging_message += "\n"

        logging_message += "Queues: \n"
        for q_key, q in self._q_dict.items():
            logging_message += f"\t{q_key}: {q.qsize()} batches (approx.)\n"

        get_logger().info(logging_message)

        # reset values
        for worker_type in self._worker_to_pid_to_num_samples.keys():
            self._worker_to_pid_to_num_samples[worker_type] = {
                pid: 0 for pid in self._worker_to_pid_to_num_samples[worker_type].keys()
            }


class ProcessingStrategyFactory:
    @staticmethod
    def get_populating_strategy(
        reader_q_key: str, logging_message_q_key: str, index_start: int, num_samples: int, batch_size: int
    ) -> PopulatingStrategy:
        return PopulatingStrategy(
            reader_q_key=reader_q_key,
            logging_message_q_key=logging_message_q_key,
            index_start=index_start,
            num_samples=num_samples,
            batch_size=batch_size,
        )

    @staticmethod
    def get_reader_strategy(
        reader_settings: TokenizationInstantiationModel.ReaderWorkerSettings.ReaderSettings,
        tokenizer_q_key: str,
        logging_message_q_key: str,
    ) -> ReadingStrategy:
        reader_type = reader_settings.reader_type
        if reader_type == LargeFileLinesReaderTypes.LOCAL:
            return ReadingStrategy(
                LargeFileLinesReaderFactory.get_local_reader,
                reader_settings.reader_args,
                tokenizer_q_key,
                logging_message_q_key,
            )
        elif reader_type == LargeFileLinesReaderTypes.GLOBAL:
            return ReadingStrategy(
                LargeFileLinesReaderFactory.get_global_reader,
                reader_settings.reader_args,
                tokenizer_q_key,
                logging_message_q_key,
            )
        else:
            raise ValueError(f"Reader type {reader_type} is not supported.")

    def get_tokenizer_strategy(
        tokenizer_settings: TokenizationInstantiationModel.TokenizerWorkerSettings.TokenizerSettings,
        writer_q_key: str,
        logging_message_q_key: str,
    ) -> TokenizingStrategy:
        tokenizing_strategy = TokenizingStrategy(
            ti_settings=tokenizer_settings.tokenizer_instantiation_settings,
            eod_token=tokenizer_settings.eod_token,
            jq_pattern=tokenizer_settings.jq_pattern,
            writer_q_key=writer_q_key,
            logging_message_q_key=logging_message_q_key,
        )
        return tokenizing_strategy

    def get_writing_strategy(
        ww_settings: TokenizationInstantiationModel.WriterWorkerSettings,
        logging_message_q_key: str,
    ) -> WritingStrategy:
        writing_strategy = WritingStrategy(
            dst_path=ww_settings.dst_path,
            index_start=ww_settings.index_start,
            logging_message_q_key=logging_message_q_key,
        )
        return writing_strategy

    def get_progress_logging_strategy(
        logging_interval: int,
        total_num_samples: int,
        q_dict: dict[str, mp.Queue],
    ) -> ProgressLoggingStrategy:
        return ProgressLoggingStrategy(
            logging_interval=logging_interval,
            total_num_samples=total_num_samples,
            q_dict=q_dict,
        )

    @staticmethod
    def get_process_queues(
        reader_q_maxsize: int, tokenizer_q_maxsize: int, writer_q_maxsize
    ) -> tuple[mp.Queue, mp.Queue, mp.Queue, mp.Queue]:
        reader_q = mp.Queue(maxsize=reader_q_maxsize)  # containes line_ids to be read
        tokenizer_q = mp.Queue(maxsize=tokenizer_q_maxsize)  # contains (line_id, line) pairs to be tokenized
        writer_q = mp.Queue(maxsize=writer_q_maxsize)  # contains (line_id, tokenized_line) to be written to disc
        logging_message_q = mp.Queue()
        return reader_q, tokenizer_q, writer_q, logging_message_q
