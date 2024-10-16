from dataclasses import field
from typing import Dict, List

import torch
from pydantic import BaseModel

from modalities.batch import DatasetBatch
from modalities.models.gpt2.collator import CollateFnIF


class AnyMALCollateFnConfig(BaseModel):
    sample_keys: List[str]
    text_sample_key: str
    text_target_key: str


class AnyMALCollatorFn(CollateFnIF):
    def __init__(self, sample_keys: List[str], text_sample_key: str, text_target_key: str):
        self.device: torch.device = field(default_factory=lambda: torch.device("cpu"))
        if text_sample_key not in sample_keys:
            raise ValueError(f"{text_sample_key} is not part of sample keys {sample_keys}")
        self.sample_keys = sample_keys  # e.g. ['images', 'input_ids']
        self.text_sample_key = text_sample_key  # input_ids
        self.text_target_key = text_target_key  # target_ids

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        unisamples = {sample_key: [] for sample_key in self.sample_keys}
        multisamples = {sample_key: [] for sample_key in self.sample_keys}
        uniindices = []
        multiindices = []
        ## gather samples which are multimodal and unimodal separately
        for idx, sample in enumerate(batch):
            if "images" in sample:
                for sample_key in self.sample_keys:
                    if isinstance(sample[sample_key], list):
                        multisamples[sample_key].append(torch.tensor(sample[sample_key]))
                    else:
                        multisamples[sample_key].append(sample[sample_key])
                multiindices.append(idx)
            else:
                for sample_key in self.sample_keys:
                    if sample_key != "images":
                        if isinstance(sample[sample_key], list):
                            unisamples[sample_key].append(torch.tensor(sample[sample_key]))
                        else:
                            unisamples[sample_key].append(sample[sample_key])
                uniindices.append(idx)

        mixed_indices = multiindices + uniindices
        samples = {}
        for sample_key in self.sample_keys:
            if len(uniindices) and sample_key != "images":
                unisamples[sample_key] = torch.stack(unisamples[sample_key])
            if len(multiindices):
                multisamples[sample_key] = torch.stack(multisamples[sample_key])
            # stack all images together
            if sample_key == "images":
                if len(multiindices):
                    samples[sample_key] = multisamples[sample_key]
            # stack the remaining sample keys as multimodal then unimodal
            elif len(uniindices) and len(multiindices):
                samples[sample_key] = torch.vstack((multisamples[sample_key], unisamples[sample_key]))
            # if only unimodal
            elif len(uniindices):
                samples[sample_key] = unisamples[sample_key]
            # if only multimodal
            elif len(multiindices):
                samples[sample_key] = multisamples[sample_key]


        targets = {}
        targets[self.text_target_key] = torch.stack([torch.tensor(d["labels"]) for d in batch])
        targets[self.text_target_key] = targets[self.text_target_key][mixed_indices]

        # if we only have unimodal samples, then the size of the labels (and attn mask later)
        # should be adjusted (since the seq length is smaller without the modality encodings)
        if len(multiindices) == 0:
            sample_size = samples[self.text_sample_key].shape[1]
            targets[self.text_target_key] = targets[self.text_target_key][:, :sample_size]


        if "attention_mask" in batch[0]:
            # all text tokens should be part of the target
            samples["attention_mask"] = torch.stack([torch.tensor(d["attention_mask"]) for d in batch])
            samples["attention_mask"] = samples["attention_mask"][mixed_indices]
            if len(multiindices) == 0:
                samples["attention_mask"] = samples["attention_mask"][:, :sample_size]
            samples["attention_mask"] = samples["attention_mask"][:, :-1]

        samples[self.text_sample_key] = samples[self.text_sample_key][:, :-1]
        targets[self.text_target_key] = targets[self.text_target_key][:, 1:]

        return DatasetBatch(targets=targets, samples=samples)
