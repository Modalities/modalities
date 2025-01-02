from dataclasses import dataclass
from enum import Enum
import math
import multiprocessing as mp
import os
import pickle
import time
import traceback
from typing import Any, Callable, Type
import warnings
from io import BufferedWriter
from pathlib import Path
from multiprocessing.synchronize import Event
from data_quality_ablations.utils.logging import get_logger
import jq
from modalities.config.instantiation_models import PackedDatasetComponentsInstantiationModel
from modalities.dataloader.preprocessing.tokenization.embedded_stream_data import EmbeddedStreamData
from modalities.exceptions import EmptySampleError
from pydantic import BaseModel
from tqdm import tqdm
import queue

from modalities.dataloader.preprocessing.tokenization.large_file_lines_reader import (
    BaseReader,
    LargeFileLinesReaderFactory,
    LargeFileLinesReaderTypes,
    Sample,
)
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


class ReaderWorker(mp.Process):
    def __init__(
        self,
        reader_type: Type[BaseReader],
        reader_args: BaseModel,
        reader_q: mp.Queue,
        tokenizer_q: mp.Queue,
        logging_message_q: mp.Queue,
        process_id: int,
        stop_event: Event,
    ):
        super().__init__()
        self._reader_q = reader_q
        self._tokenizer_q = tokenizer_q
        self._logging_message_q = logging_message_q
        self._reader_type = reader_type
        self._reader_args = reader_args
        self._stop_event = stop_event
        self.process_id = process_id

    def run(self):
        reader = self._reader_type(**self._reader_args.model_dump())
        batch = []
        num_samples_read = 0
        while not self._stop_event.is_set():
            try:
                # we set the timout here, such that the worker can check if the stop_event is set
                item = self._reader_q.get(timeout=3)
            except queue.Empty:
                continue
            if item is None:
                print(f"Reading worker with pid {mp.current_process().pid} exiting, Read {num_samples_read} samples")
                break
            sample_id, batch_size = item


            batch: list[Sample] = [reader[sample_id + i] for i in range(batch_size)]
            self._tokenizer_q.put(batch)
            self._logging_message_q.put(ProgressMessage(WorkerTypes.READER, self.process_id, len(batch)))
            num_samples_read += len(batch)

        if not self._stop_event.is_set():
            # add the remaining samples
            if len(batch) > 0:
                self._tokenizer_q.put(batch)
                self._logging_message_q.put(ProgressMessage(WorkerTypes.READER, self.process_id, len(batch)))



class TokenizerWorker(mp.Process):
    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        eod_token: str,
        token_size_in_bytes: int,
        tokenizer_q: mp.Queue,
        logging_message_q: mp.Queue,
        writer_q: mp.Queue,
        jq_pattern: str,
        process_id: int,
        stop_event: Event,
    ):
        super().__init__()
        self._jq_filter = jq.compile(jq_pattern)
        self.tokenizer = tokenizer
        self.eod_token = eod_token
        self._token_size_in_bytes = token_size_in_bytes
        encoded_eod_token = self.tokenizer.get_token_id(self.eod_token)
        self._encoded_eos_token_as_bytes = self._encoded_token_to_bytes(encoded_eod_token)
        self._tokenizer_q = tokenizer_q
        self._writer_q = writer_q
        self._logging_message_q = logging_message_q
        self._process_id = process_id
        self._stop_event = stop_event

    def run(self):
        # Process the lines in a batch and put the processed samples into the writer_q.

        while not self._stop_event.is_set():
            try:
                batch: list[Sample] = self._tokenizer_q.get(timeout=10)
            except queue.Empty:
                continue
            if batch is None:
                break

            try:
                batch_processed = []
                for sample in batch:
                    processed_line = self._process_line(sample.content_raw)
                    sample.content_tokenized = processed_line
                    batch_processed.append(sample)
                self._writer_q.put(batch_processed)
                self._logging_message_q.put(ProgressMessage(WorkerTypes.TOKENIZER, self._process_id, len(batch)))
            except EmptySampleError:
                warnings.warn(
                    f"Encountered empty sample in line {sample.shuffled_line_id} in file {sample.raw_data_path} within process {self._process_id}"
                )
            except Exception as exception:
                warnings.warn(
                    f"Could not process line {sample.shuffled_line_id} in file {sample.raw_data_path} within process {self._process_id}. "
                    f"Raised the following error: {exception=}"
                )
                traceback.print_exc()

    def _process_line(self, line: str) -> bytes:
        # extracts the text via the jq_filter and applies tokenization to the extract text
        jq_retrieved_text = self._jq_filter.input_text(line).first()
        if jq_retrieved_text is None:
            raise ValueError(f"jq was not able to find anything using the expression: {self._jq_filter}")
        tokens = self.tokenizer.tokenize(jq_retrieved_text)
        if len(tokens) == 0:
            raise EmptySampleError("Received empty sample...")
        return b"".join(map(self._encoded_token_to_bytes, tokens)) + self._encoded_eos_token_as_bytes

    def _encoded_token_to_bytes(self, encoded_token: int) -> bytes:
        """
        Converts an encoded token to its byte representaion.

        Args:
            encoded_token (int): The encoded token to be converted.

        Returns:
            bytes: The byte representation of the token.

        """
        return encoded_token.to_bytes(self._token_size_in_bytes, byteorder="little", signed=False)


class WriterWorker(mp.Process):
    def __init__(
        self, token_size_in_bytes: int, writer_q: mp.Queue, logging_message_q: mp.Queue, dst_path: Path, stop_event: Event, index_start: int, 
        process_id: int
    ):
        super().__init__()
        self._token_size_in_bytes = token_size_in_bytes
        self._dst_path = dst_path
        self._writer_q = writer_q
        self._logging_message_q = logging_message_q
        self._stop_event = stop_event
        self._index_start = index_start
        self.process_id = process_id

    def run(self):
        index_list = []
        if not self._dst_path.parent.exists():
            self._dst_path.parent.mkdir(parents=True, exist_ok=True)
        with self._dst_path.open("wb") as f:
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
            prev_line_id = self._index_start - 1
            batch_dict = {}
            while not self._stop_event.is_set():
                try:
                    batch: list[Sample] = self._writer_q.get(timeout=3)
                except queue.Empty:
                    continue
                if batch is None:
                    break
                line_id = batch[0].incremental_line_id
                batch_dict[line_id] = batch

                while prev_line_id + 1 in batch_dict:
                    batch = batch_dict.pop(prev_line_id + 1)
                    prev_line_id, curr_offset = WriterWorker._write_batch(
                        batch, prev_line_id, curr_offset, index_list, f
                    )
                    self._logging_message_q.put(ProgressMessage(WorkerTypes.WRITER, self.process_id, len(batch)))

            # write index
            f.write(pickle.dumps(index_list))
        if not self._stop_event.is_set() and len(index_list) > 0 and len(batch_dict) == 0:
            self._update_data_length_in_pre_allocated_header(self._dst_path, index_list)
        else:
            # if the process was stopped due to a stop event or the index list is empty, we remove the file
            get_logger(name="main").warning(f"Removing file {self._dst_path} due to empty index list or stop event or non-empty batch_dict. " 
                                            f"stop_event: {self._stop_event.is_set()}, index_list: {len(index_list)}, batch_dict: {batch_dict.keys()}")
            os.remove(self._dst_path)

    # writes a batch received from the writer_q to the destination file
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


class WorkerTypes(Enum):
    READER = "READER"
    TOKENIZER = "TOKENIZER"
    WRITER = "WRITER"

@dataclass
class ProgressMessage:
    worker_type: WorkerTypes
    process_id: int
    num_samples: int


class ProgressLoggingWorker(mp.Process):
    def __init__(self, logging_message_q: mp.Queue, logging_interval: int, reader_q: mp.Queue, tokenizer_q: mp.Queue, writer_q: mp.Queue, total_num_samples: int, stop_event: Event):
        super().__init__()
        self._logging_message_q = logging_message_q
        self._logging_interval = logging_interval
        self._reader_q = reader_q
        self._tokenizer_q = tokenizer_q
        self._writer_q = writer_q
        self._stop_event = stop_event
        self._worker_to_pid_to_num_samples: dict[WorkerTypes, dict[int, int]] = {}
        
        self._total_num_samples = total_num_samples
        self._worker_type_to_processed_num_samples = {worker_type: 0 for worker_type in WorkerTypes}

    def _add_progress_message(self, progress_message: ProgressMessage):
        if progress_message.worker_type not in self._worker_to_pid_to_num_samples:
            self._worker_to_pid_to_num_samples[progress_message.worker_type] = {}

        if progress_message.process_id not in self._worker_to_pid_to_num_samples[progress_message.worker_type]:
            self._worker_to_pid_to_num_samples[progress_message.worker_type][progress_message.process_id] = 0
        
        self._worker_to_pid_to_num_samples[progress_message.worker_type][progress_message.process_id] += progress_message.num_samples
        self._worker_type_to_processed_num_samples[progress_message.worker_type] += progress_message.num_samples


    def _log_and_reset(self, passed_time: int):
        logging_message = f"\n==================Progress report (last {passed_time}s) ==================\n"
        
        logging_message += f"Total progress: \n"
        for worker_type, processed_num_samples in self._worker_type_to_processed_num_samples.items():
            logging_message += f"\t{worker_type.name}: {processed_num_samples}/{self._total_num_samples} samples ({processed_num_samples/self._total_num_samples*100}%)\n"


        logging_message += "\n"
        logging_message += f"Aggregated Throughput: \n"

        for worker_type, pid_to_num_samples in self._worker_to_pid_to_num_samples.items():    
            total_samples = sum(pid_to_num_samples.values())
            logging_message += f"\t{worker_type.name} workers: {total_samples/passed_time} samples/s.\n"
        logging_message += "\n"
        logging_message += f"Worker Throughput: \n"
        for worker_type, pid_to_num_samples in self._worker_to_pid_to_num_samples.items():
            logging_message += f"{worker_type.name} workers:\n"
            for pid, num_samples in pid_to_num_samples.items():
                logging_message += f"\t{worker_type.name} {pid}: {num_samples/passed_time} samples/s.\n"
            logging_message += "\n"
        logging_message += "\n"
        
        logging_message += "Queues: \n"
        logging_message += f"\tReader queue: {self._reader_q.qsize()} batches (approx.)\n"
        logging_message += f"\tTokenizer queue: {self._tokenizer_q.qsize()} batches (approx.)\n"
        logging_message += f"\tWriter queue: {self._writer_q.qsize()} batches (approx.)\n"

        get_logger().info(logging_message)

        # reset values
        for worker_type in self._worker_to_pid_to_num_samples.keys():
            self._worker_to_pid_to_num_samples[worker_type] = {pid: 0 for pid in self._worker_to_pid_to_num_samples[worker_type].keys()}


    def run(self):
        last_logged = time.time()
        last_step = False
        while not self._stop_event.is_set():
            try:
                progress_message: ProgressMessage = self._logging_message_q.get(timeout=1)
                if progress_message is None:
                    last_step = True
                    break
                self._add_progress_message(progress_message)
            except queue.Empty:
                continue
            finally:
                passed_time = time.time() - last_logged
                if passed_time > self._logging_interval or last_step:
                    self._log_and_reset(passed_time)
                    last_logged = time.time()


class ProcessFactory:

    @staticmethod
    def get_reader_workers(
        rw_settings: PackedDatasetComponentsInstantiationModel.ReaderWorkerSettings,
        reader_q: mp.Queue,
        tokenizer_q: mp.Queue,
        logging_message_q: mp.Queue,
        stop_event: Event,
    ) -> list[tuple[Type[Callable], BaseModel]]:
        # create readers
        reader_type = rw_settings.reader_settings.reader_type
        if reader_type == LargeFileLinesReaderTypes.LOCAL:
            readers = [
                (LargeFileLinesReaderFactory.get_local_reader, rw_settings.reader_settings.reader_args)
                for _ in range(rw_settings.num_reader_processes)
            ]

        elif reader_type == LargeFileLinesReaderTypes.GLOBAL:
            readers = [
                (LargeFileLinesReaderFactory.get_global_reader, rw_settings.reader_settings.reader_args)
                for _ in range(rw_settings.num_reader_processes)
            ]
        else:
            raise ValueError(f"Reader type {reader_type} is not supported.")

        # create reader workers
        reader_workers = [
            ReaderWorker(
                reader_type= reader_type,
                reader_args = reader_args,
                reader_q=reader_q,
                tokenizer_q=tokenizer_q,
                logging_message_q=logging_message_q,
                stop_event=stop_event,
                process_id=pid,
  
            )
            for pid, (reader_type, reader_args) in enumerate(readers)
        ]

        return reader_workers

    def get_tokenizer_workers(
        tokenizer_q: mp.Queue,
        writer_q: mp.Queue,
        logging_message_q: mp.Queue,
        token_size_in_bytes: int,
        tw_settings: PackedDatasetComponentsInstantiationModel.TokenizerWorkerSettings,
        stop_event: Event,
    ) -> list[TokenizerWorker]:
        tokenizer_settings = tw_settings.tokenizer_settings
        tokenizer_workers = [
            TokenizerWorker(
                process_id=i,
                stop_event=stop_event,
                tokenizer_q=tokenizer_q,
                writer_q=writer_q,
                logging_message_q=logging_message_q,
                token_size_in_bytes=token_size_in_bytes,
                **tokenizer_settings.model_dump(),
            )
            for i in range(tw_settings.num_tokenizer_processes)
        ]
        return tokenizer_workers

    def get_writer_worker(
        writer_q: mp.Queue,
        logging_message_q: mp.Queue,
        token_size_in_bytes: int,
        ww_settings: PackedDatasetComponentsInstantiationModel.WriterWorkerSettings,
        stop_event: Event,
    ) -> WriterWorker:
        writer_worker = WriterWorker(
            writer_q=writer_q,
            logging_message_q=logging_message_q,
            token_size_in_bytes=token_size_in_bytes,
            dst_path=ww_settings.dst_path,
            index_start=ww_settings.index_start,
            stop_event=stop_event,
            process_id=0,
        )
        return writer_worker

    @staticmethod
    def get_process_queues(tokenizer_q_maxsize: int, writer_q_maxsize) -> tuple[mp.Queue, mp.Queue, mp.Queue]:
        reader_q = mp.Queue()  # containes line_ids to be read
        tokenizer_q = mp.Queue(maxsize=tokenizer_q_maxsize)  # contains (line_id, line) pairs to be tokenized
        writer_q = mp.Queue(maxsize=writer_q_maxsize)  # contains (line_id, tokenized_line) to be written to disc
        logging_message_q = mp.Queue()
        return reader_q, tokenizer_q, writer_q, logging_message_q
