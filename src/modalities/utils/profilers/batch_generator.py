from abc import ABC

import torch
from pydantic import BaseModel

from modalities.batch import DatasetBatch


class RandomDatasetBatchGeneratorConfig(BaseModel):
    vocab_size: int
    sequence_length: int
    batch_size: int


class DatasetBatchGeneratorIF(ABC):
    def get_dataset_batch(self) -> DatasetBatch:
        raise NotImplementedError


class RandomDatasetBatchGenerator(DatasetBatchGeneratorIF):
    def __init__(self, vocab_size: int, sequence_length: int, batch_size: int):
        self._vocab_size = vocab_size
        self._sequence_length = sequence_length
        self._batch_size = batch_size

    def get_dataset_batch(self) -> DatasetBatch:
        batch = DatasetBatch(
            samples={"input_ids": torch.randint(0, self._vocab_size, (self._batch_size, self._sequence_length))},
            targets={"target_ids": torch.randint(0, self._vocab_size, (self._batch_size, self._sequence_length))},
        )
        return batch
