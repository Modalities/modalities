from dataclasses import field
from typing import Dict, List

import torch
from pydantic import BaseModel

from modalities.batch import DatasetBatch
from modalities.models.gpt2.collator import CollateFnIF


class CoCaCollateFnConfig(BaseModel):
    sample_keys: List[str]
    target_keys: List[str]
    text_sample_key: str
    text_target_key: str


class CoCaCollatorFn(CollateFnIF):
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
            sample_key: torch.stack([self._prepare_sample(d[sample_key]) for d in batch])
            for sample_key in self.sample_keys
        }
        if "attention_mask" in batch[0]:
            samples["attention_mask"] = torch.stack([self._prepare_sample(d["attention_mask"]) for d in batch])

        targets = {
            target_key: torch.stack([self._prepare_sample(d[target_key]) for d in batch])
            for target_key in self.target_keys
        }

        # Create target for text input
        targets[self.text_target_key] = samples[self.text_sample_key][:, 1:].clone().detach()
        samples[self.text_sample_key] = samples[self.text_sample_key][:, :-1]

        if "attention_mask" in batch[0]:
            targets["attention_mask"] = samples["attention_mask"][:, 1:].clone().detach()
            samples["attention_mask"] = samples["attention_mask"][:, :-1]

        return DatasetBatch(targets=targets, samples=samples)

    @staticmethod
    def _prepare_sample(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)
