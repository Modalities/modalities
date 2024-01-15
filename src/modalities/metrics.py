from abc import ABC, abstractmethod

import torch

from modalities.batch import InferenceResultBatch


class Metric(ABC):
    def __init__(self, tag: str):
        self._tag = tag

    @property
    def tag(self) -> str:
        return self._tag

    @abstractmethod
    def __call__(self, result_batch: InferenceResultBatch) -> torch.Tensor:
        raise NotImplementedError
