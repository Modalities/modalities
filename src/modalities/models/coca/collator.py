from dataclasses import field
from typing import Dict, List

import torch

from modalities.batch import DatasetBatch


class CoCaCollator:
    def __init__(self, sample_keys: List[str], target_keys: List[str], text_sample_key: str, text_target_key: str):
        self.device: torch.device = field(default_factory=lambda: torch.device("cpu"))
        if text_sample_key not in sample_keys:
            raise ValueError(f"{text_sample_key} is not part of sample keys {sample_keys}")
        if text_target_key in target_keys:
            raise ValueError(
                f"{text_target_key} should not be part of target keys {target_keys}, "
                f"because {text_target_key} will generated based on {text_sample_key}"
            )
        self.sample_keys = sample_keys
        self.target_keys = target_keys
        self.text_sample_key = text_sample_key
        self.text_target_key = text_target_key

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        samples = {
            sample_key: torch.stack([torch.tensor(d[sample_key]) for d in batch]) for sample_key in self.sample_keys
        }
        targets = {
            target_key: torch.stack([torch.tensor(d[target_key]) for d in batch]) for target_key in self.target_keys
        }

        # Create target for text input
        targets[self.text_target_key] = samples[self.text_sample_key][:, 1:].clone().detach()
        samples[self.text_sample_key] = samples[self.text_sample_key][:, :-1].clone().detach()
        return DatasetBatch(targets=targets, samples=samples)
