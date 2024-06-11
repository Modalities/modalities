from abc import abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from modalities.batch import DatasetBatch, InferenceResultBatch

WeightDecayGroups = Dict[str, List[str]]


class NNModel(nn.Module):
    def __init__(self, seed: int = None, weight_decay_groups: Optional[WeightDecayGroups] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self._weight_decay_groups = weight_decay_groups if weight_decay_groups is not None else {}
        super(NNModel, self).__init__()

    @property
    def weight_decay_groups(self) -> WeightDecayGroups:
        return self._weight_decay_groups

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in self.named_parameters()}


def model_predict_batch(model: nn.Module, batch: DatasetBatch) -> InferenceResultBatch:
    forward_result = model.forward(batch.samples)
    result_batch = InferenceResultBatch(targets=batch.targets, predictions=forward_result)
    return result_batch
