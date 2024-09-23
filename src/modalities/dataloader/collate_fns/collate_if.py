from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from modalities.batch import DatasetBatch


class CollateFnIF(ABC):
    """CollateFnIF class to define a collate function interface."""

    @abstractmethod
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch:
        """
        Process a batch of data.

        Args:
            batch (List[Dict[str, torch.Tensor]]): A list of dictionaries containing tensors.

        Returns:
            DatasetBatch: The processed batch of data.

        Raises:
            NotImplementedError: This abstract method should be implemented in a subclass.
        """
        raise NotImplementedError
