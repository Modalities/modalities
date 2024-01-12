from dataclasses import field
from typing import Dict, List

import torch

from modalities.batch import DatasetBatch


class GPT2LLMCollator:
    def __init__(self, sample_key: str, target_key: str):
        self.device: torch.device = field(default_factory=lambda: torch.device("cpu"))
        self.sample_key = sample_key
        self.target_key = target_key

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        sample_tensor = torch.stack([torch.tensor(d[self.sample_key]) for d in batch])
        samples = {self.sample_key: sample_tensor[:, :-1]}
        targets = {self.target_key: sample_tensor[:, 1:]}

        return DatasetBatch(targets=targets, samples=samples)
