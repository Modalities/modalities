from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from modalities.batch import DatasetBatch


class CollateFnIF(ABC):
    @abstractmethod
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        raise NotImplementedError


class GPT2LLMCollateFn(CollateFnIF):
    def __init__(self, sample_key: str, target_key: str):
        self.sample_key = sample_key
        self.target_key = target_key

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        sample_tensor = torch.stack([torch.tensor(d[self.sample_key]) for d in batch])
        samples = {self.sample_key: sample_tensor[:, :-1]}
        targets = {self.target_key: sample_tensor[:, 1:]}

        return DatasetBatch(targets=targets, samples=samples)
