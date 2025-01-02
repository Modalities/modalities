import multiprocessing as mp
import time


from modalities.dataloader.preprocessing.tokenization.tokenization_processes import (
    ProgressLoggingWorker,
    ReaderWorker,
    TokenizerWorker,
    WriterWorker,
)
from modalities.utils.env_variables import temporary_env_var
from modalities.utils.logging import get_logger
import tqdm
import time




class PackedDataGenerator:
    """Reads in a JSONL file and the corresponding index file and packs the dataset for LLM training."""

    def __init__(
        self,
        reader_workers: list[ReaderWorker],
        tokenizer_workers: list[TokenizerWorker],
        writer_worker: WriterWorker,
        progress_logging_worker: ProgressLoggingWorker,
        reader_q: mp.Queue,
        tokenizer_q: mp.Queue,
        writer_q: mp.Queue,
        logging_message_q: mp.Queue,
        index_start: int,
        num_samples: int,
        batch_size: int,
    ):
        self.reader_workers = reader_workers
        self.tokenizer_workers = tokenizer_workers
        self.writer_worker = writer_worker
        self.progress_logging_worker = progress_logging_worker
        self.reader_q = reader_q
        self.tokenizer_q = tokenizer_q
        self.writer_q = writer_q
        self.logging_message_q = logging_message_q
        self._index_start = index_start
        self._num_samples = num_samples
        self.batch_size = batch_size
        self._exception_buffer = []

        if num_samples == -1:
            # TODO accessing the reader directly is not nice, but we need to know the total number of samples
            total_num_samples = len(self.reader_workers[0]._reader)
            num_samples = total_num_samples - index_start

    def run(self):
        # Not setting TOKENIZERS_PARALLELISM to false can cause deadlocks when using hf's "FastTokenizers". See also:
        # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/67254879#67254879
        with temporary_env_var("TOKENIZERS_PARALLELISM", "false"):
            start_time = time.time()
            # populate the reader queue with the sample_ids that we want to tokenize
            self._populate_reader_q(
                index_start=self._index_start,
                num_samples=self._num_samples,
                num_reader_processes=len(self.reader_workers),
            )
            
            # start the progress logging worker
            self.progress_logging_worker.start()

            # start the reader proceseses
            for reader_worker in tqdm.tqdm(self.reader_workers, desc="Starting reader workers"):
                reader_worker.start()

            # start the tokenizer processes
            for tokenizer_worker in tqdm.tqdm(self.tokenizer_workers, desc="Starting tokenizer workers"):
                tokenizer_worker.start()

            # start the writer process
            self.writer_worker.start()

            # wait for all processes to finish
            for reader_worker in tqdm.tqdm(self.reader_workers, desc="Stopping for reader workers"):
                reader_worker.join()

            # stop the tokenizer processes
            for _ in self.tokenizer_workers:
                self.tokenizer_q.put(None)
            for tokenizer_worker in tqdm.tqdm(self.tokenizer_workers, desc="Stopping tokenizer workers"):
                tokenizer_worker.join()

            # stop the writer process
            get_logger().info("Stopping writer worker.")
            self.writer_q.put(None)
            self.writer_worker.join()
            
            # stop the logging worker process
            get_logger().info("Stopping progress logging worker.")
            self.logging_message_q.put(None)
            self.progress_logging_worker.join()

            end_time = time.time()
            get_logger().info(f"Tokenization took {end_time - start_time} seconds.")
            
            if self._exception_buffer:
                raise self._exception_buffer[0]


    def _populate_reader_q(self, index_start: int, num_samples: int, num_reader_processes: int):
        # populate the reader queue with the line_ids that we want to tokenize
        
        for i in tqdm.tqdm(range(index_start, index_start + num_samples, self.batch_size), desc="Filling up reader queue with line ids"):
            self.reader_q.put((i, self.batch_size))
        for i in range(num_reader_processes):
            self.reader_q.put(None)
