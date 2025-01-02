import json
import os
import pickle as pkl
import queue
import threading
import time
from pathlib import Path
import jq

from modalities.utils.logging import get_logger
from tqdm import tqdm


class IndexGenerator:
    def __init__(self, src_file: Path, drop_faulty_entries: bool = False, jq_pattern: str = ".text"):
        """
        Initializes an IndexGenerator object.
        Reads a JSONL file as a binary file, and iterates through it character by character.
        It builds the sample index by tracking the start and end positions of each JSON sample
        based on the positions of newline (\n) characters.

        Args:
            src_file (Path): Path to a jsonl-file.
            drop_faulty_entries (bool): Allow broken json entries in `src_file` by just skipping them.
                Otherwise, the index generation fails with an exception.

        Returns:
            None
        """
        self.src_file = src_file
        self.jq_pattern = jq_pattern
        self.drop_faulty_entries = drop_faulty_entries
        with self.src_file.open(mode="rb") as fin:
            # Move the cursor to the end of the file
            fin.seek(0, os.SEEK_END)
            # Get number of bytes in the file
            self._total_num_bytes = fin.tell()
        self._queue_of_raw_lines = queue.Queue()
        self._index_map = []
        self._exception_buffer = []

    def create_index(self, target_path_for_index_file: Path):
        """
        Creates an index file where each item in the index represents the start and length of a
        JSON document within a JSONL file.

        Args:
            target_path_for_index_file (Path): The path where the index file will be created.

        Raises:
            Exception: If an exception occurs during the indexing process.

        Returns:
            None
        """
        start_time = time.time()
        self._exception_buffer = []
        reader = threading.Thread(target=self._reader_thread)
        reader.start()
        processor = threading.Thread(target=self._indexer_thread)
        processor.start()
        reader.join()
        processor.join()
        if self._exception_buffer:
            get_logger(name="main").warning(
                f"Index creation failed for {target_path_for_index_file}. Exception buffer: {self._exception_buffer}"
            )
            raise self._exception_buffer[0]

        if len(self._index_map) == 0:
            get_logger(name="main").warning(f"Could not create index! No entries found in {self.src_file}")
        else:
            end_time = time.time()
            get_logger(name="main").info(
                f"Created index {target_path_for_index_file} of length {len(self._index_map)} "
                f"at {len(self._index_map) / (end_time - start_time)} iterations/s."
            )
            target_path_for_index_file.write_bytes(pkl.dumps(self._index_map))
            get_logger(name="main").info(f"Wrote index {target_path_for_index_file} to disc.")

    def _indexer_thread(self):
        # This method is responsible for indexing the lines in the queue and parsing them as JSON.
        # It iterates over the lines in the queue and checks if each line is a valid JSON. If a line is valid,
        # it appends the line's start index and length to the index map. If a line is not valid, it either
        # skips the line or raises a ValueError, depending on the value of `drop_faulty_entries`.

        def queue_generator():
            # Generator function that continuously yields lines
            # (i.e. JSON documents) from a queue until None is encountered.

            while True:
                line = self._queue_of_raw_lines.get()
                if line is None:
                    break
                yield line

        def parse_line_as_json(line_id: int, line_start_byte_pos: int, line: bytes, jq_filter):
            # Parses a line as JSON and appends the sample index, i.e.,
            # the line start index and length to the index map.
            # If the line is faulty and `drop_faulty_entries` is set to True, a warning is issued.
            line_string = line.decode("utf-8")
            jq_retrieved_text = jq_filter.input_text(line_string).first()
            if jq_retrieved_text is not None:
                if len(jq_retrieved_text) > 0:
                    self._index_map.append((line_start_byte_pos, len(line)))
                else:
                    get_logger(name="main").warning(f'Faulty line {line_id} (no text) in {str(self.src_file)}, skipping...')
            else:
                if self.drop_faulty_entries:
                    get_logger(name="main").warning(f'Faulty line {line_id} (parsing error) in {str(self.src_file)}, skipping...')
                else:
                    get_logger(name="main").warning(f'Faulty line {line_id} (parsing error), stopping...')
                    err = ValueError(f'Faulty line "{line} in {str(self.src_file)}')
                    self._exception_buffer.append(err)

        jq_filter = jq.compile(self.jq_pattern)
        self._index_map = []
        for line_id, line_start_byte_pos, line in tqdm(queue_generator(), desc="Processed Lines", disable=True):
            if self._check_for_parallel_errors():
                return
            parse_line_as_json(line_id, line_start_byte_pos, line, jq_filter)

    def _reader_thread(self):
        # Reads lines from the source file and puts them into a queue.
        # This method is executed in a separate thread. It reads lines from the source file until
        # the end of the file is reached. Each line is put into a queue along with its cursor position. If any
        # errors are detected, the method returns immediately.
        get_logger(name="main").info(f"Reading the jsonl file {self.src_file}...")

        num_read_documents = 0
        with open(self.src_file, "rb") as fin:
            while True:
                cursor = fin.tell()
                line = fin.readline()
                if self._check_for_parallel_errors():
                    return
                if fin.tell() == self._total_num_bytes:
                    if line.endswith(b"\n"):
                        line = line[:-1]
                    self._queue_of_raw_lines.put((num_read_documents, cursor, line))
                    num_read_documents += 1
                    break
                line_without_newline_char = line[:-1]
                self._queue_of_raw_lines.put((num_read_documents, cursor, line_without_newline_char))
                num_read_documents += 1
        get_logger(name="main").info(f"Finished reading the jsonl file {self.src_file} (read {num_read_documents}).")
        self._queue_of_raw_lines.put(None)

    def _check_for_parallel_errors(self) -> bool:
        # Checks if there are any errors in the exception buffer.
        return bool(self._exception_buffer)
