import json
import os
import pickle as pkl
import queue
import threading
import warnings
from pathlib import Path

from tqdm import tqdm


class IndexGenerator:
    def __init__(self, src_file: Path, chunksize: int = 4096, drop_faulty_entries: bool = False):
        """
        Reads in a JSON file as a binary file, iterates character by character und builds up
        the sample index (char-wise start and end position for each JSON sample) via "\n" character positions.

        :param src_file: Path to a jsonl-file.
        :param chunksize: defines the size of byte chunks that are processed via a producer-consumer approach.
                          The producer reads chunks from the `src_file`, while the consumer creates index entries.
        :param drop_faulty_entries: Allow broken json entries in `src_file` by just skipping them.
                                    Otherwise, the index generation fails with an exception.
        """
        self.src_file = src_file
        self.chunksize = chunksize
        self.drop_faulty_entries = drop_faulty_entries
        with self.src_file.open(mode="r") as fin:
            # Move the cursor to the end of the file
            fin.seek(0, os.SEEK_END)
            # Get number of characters in the file
            self._total_num_chars = fin.tell()
        self.num_chunks = self._total_num_chars // self.chunksize
        self._queue_of_raw_lines = queue.Queue()
        self._index_map = []
        self._exception_buffer = []

    def create_index(self, target_path_for_index_file: Path):
        self._exception_buffer = []
        reader = threading.Thread(target=self._reader_thread)
        reader.start()
        processor = threading.Thread(target=self._indexer_thread)
        processor.start()
        reader.join()
        processor.join()
        if self._exception_buffer:
            raise self._exception_buffer[0]
        print(f"Created index of length {len(self._index_map)}")
        target_path_for_index_file.write_bytes(pkl.dumps(self._index_map))

    def _indexer_thread(self):
        def queue_generator():
            while True:
                line = self._queue_of_raw_lines.get()
                if line is None:
                    break
                yield line

        def parse_line_as_json(line_start_idx: int, line: str):
            try:  # check if line is a valid json
                json.loads(line)
                self._index_map.append((line_start_idx, len(line)))
            except Exception as low_level_err:
                if self.drop_faulty_entries:
                    warnings.warn(f'faulty line "{line}", skipping...')
                else:
                    err = ValueError(f'faulty line "{line}", skipping...')
                    err.__cause__ = low_level_err
                    self._exception_buffer.append(err)

        self._index_map = []
        for line_start_idx, line in tqdm(queue_generator(), desc="Processed Lines"):
            if self._check_for_parallel_errors():
                return
            parse_line_as_json(line_start_idx, line)

    def _reader_thread(self):
        with open(self.src_file, "r") as fin:
            while True:
                cursor = fin.tell()
                line = fin.readline()
                if self._check_for_parallel_errors():
                    return
                if fin.tell() == self._total_num_chars:
                    self._queue_of_raw_lines.put((cursor, line))
                    break
                line_without_newline_char = line[:-1]
                self._queue_of_raw_lines.put((cursor, line_without_newline_char))
        self._queue_of_raw_lines.put(None)

    def _check_for_parallel_errors(self) -> bool:
        return bool(self._exception_buffer)
