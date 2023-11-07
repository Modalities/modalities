import json
import os
import pickle as pkl
import queue
import threading
from pathlib import Path

import numpy as np
from tqdm import tqdm


# TODO: benchmark against pyspark
class IndexGenerator:
    def __init__(self, src: Path, chunksize: int = 4096):
        self.src = Path(src)
        self.chunksize = chunksize
        with self.src.open(mode="r", encoding="utf-8") as fin:
            fin.seek(0, os.SEEK_END)
            char_num = fin.tell()
        self.chunks = char_num // self.chunksize
        self.reminder = char_num % self.chunksize
        self.chunk_queue = queue.Queue()
        self.index_map = []

    def run(self, dst: Path):
        reader = threading.Thread(target=self._reader_thread)
        reader.start()
        processor = threading.Thread(target=self._indexer_thread)
        processor.start()
        reader.join()
        processor.join()
        print(f"Created index of length {len(self.index_map)}")
        dst.write_bytes(pkl.dumps(self.index_map))

    def _indexer_thread(self):
        def queue_generator():
            while True:
                chunk = self.chunk_queue.get()
                if chunk is None:
                    break
                yield chunk

        self.index_map = []
        last_index = 0
        for chunk_idx, chunk in tqdm(enumerate(queue_generator()), desc="Processed Chunks", total=self.chunks):
            for char_index, c in enumerate(chunk):
                curr_index = chunk_idx * self.chunksize + char_index
                if c == ord("\n"):
                    segment_len = curr_index - last_index
                    try:  # check if line is a valid json
                        string = (
                            np.memmap(self.src, mode="r", offset=last_index, shape=(segment_len,)).view("S1").tolist()
                        )
                        string = [c.decode("iso-8859-1") for c in string]
                        string = "".join(string)
                        _ = json.loads(string)
                        self.index_map.append((last_index, segment_len))
                    except Exception:
                        print(f"\nfaulty line at {last_index}-{curr_index}, skipping...")
                    last_index = curr_index + 1
        # TODO: implement proper handling of remaining (not full) segment

    def _reader_thread(self):
        with open(self.src, "rb") as fin:
            while True:
                chunk = fin.read(self.chunksize)
                if not chunk:
                    break
                self.chunk_queue.put(chunk)
        self.chunk_queue.put(None)
