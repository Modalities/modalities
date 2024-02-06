import json
import os
import pickle as pkl
import queue
import threading
import warnings
from pathlib import Path

from tqdm import tqdm

from modalities.constants import DEFAULT_ENCODING


# TODO: benchmark against pyspark
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
        with self.src_file.open(mode="r", encoding="utf-8") as fin:
            fin.seek(0, os.SEEK_END)
            num_chars = fin.tell()
        self.num_chunks = num_chars // self.chunksize
        self.reminder = num_chars % self.chunksize
        self._chunk_queue = queue.Queue()
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
                chunk = self._chunk_queue.get()
                if chunk is None:
                    break
                yield chunk

        def process_line(last_index: int, curr_index: int):
            segment_len = curr_index - last_index
            try:  # check if line is a valid json
                f.seek(last_index)
                decoded_line = f.read(segment_len)
                json.loads(decoded_line)
                self._index_map.append((last_index, segment_len))
            except Exception as low_level_err:
                if self.drop_faulty_entries:
                    warnings.warn(f"faulty line at {last_index}-{curr_index}, skipping...")
                else:
                    warnings.warn(f"faulty line: {decoded_line=}")
                    err = ValueError(f"faulty line at {last_index}-{curr_index}")
                    err.__cause__ = low_level_err
                    self._exception_buffer.append(err)

        f = self.src_file.open(encoding=DEFAULT_ENCODING)
        self._index_map = []
        last_index = 0
        for chunk_idx, chunk in tqdm(enumerate(queue_generator()), desc="Processed Chunks", total=self.num_chunks):
            for char_index, c in enumerate(chunk):
                curr_index = chunk_idx * self.chunksize + char_index
                if c == "\n":
                    process_line(last_index, curr_index)
                    last_index = curr_index + 1
        # prevents automatically added "\n"-chars at the end of files getting interpreted as own sample
        if curr_index >= last_index:
            process_line(last_index, curr_index + 1)

    def _reader_thread(self):
        with open(self.src_file, "r", encoding=DEFAULT_ENCODING) as fin:
            while True:
                chunk = fin.read(self.chunksize)
                if self._exception_buffer:
                    raise RuntimeError(
                        "Exception found in exception buffer. Probably the indexer thread ran into an error..."
                    )
                if not chunk:
                    break
                self._chunk_queue.put(chunk)
        self._chunk_queue.put(None)
