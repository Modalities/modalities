from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from modalities.batch import DatasetBatch


class CollateFnIF(ABC):
    @abstractmethod
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        raise NotImplementedError
