from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torch.distributed.device_mesh import DeviceMesh

from modalities.batch import DatasetBatch


class CollateFnIF(ABC):
    @abstractmethod
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        raise NotImplementedError


class GPT2LLMCollateFn(CollateFnIF):
    def __init__(self, sample_key: str, target_key: str, device_mesh: DeviceMesh):
        self.sample_key = sample_key
        self.target_key = target_key
        self.device_mesh = device_mesh

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        sample_tensor = torch.stack([torch.tensor(d[self.sample_key]) for d in batch])
        sample_tensor_distributed = (
            sample_tensor.long()
        )  # torch.LongTensor(sample_tensor) #distribute_tensor(sample_tensor, self.device_mesh, [Shard(dim=0)])

        samples = {self.sample_key: sample_tensor_distributed[:, :-1]}
        targets = {self.target_key: sample_tensor_distributed[:, 1:]}

        return DatasetBatch(targets=targets, samples=samples)
