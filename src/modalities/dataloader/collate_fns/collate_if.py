from abc import ABC, abstractmethod

import torch

from modalities.batch import DatasetBatch


class CollatorIF(ABC):
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> DatasetBatch:
        """
        Process a batch of data.

        Args:
            batch (list[dict[str, torch.Tensor]]): A list of dictionaries containing 1-dim tensors.

        Returns:
            DatasetBatch: The processed batch of data.

        Raises:
            NotImplementedError: This abstract method should be implemented in a subclass.
        """
        raise NotImplementedError


class CollateFnIF(ABC):
    """CollateFnIF class to define a collate function interface."""

    @abstractmethod
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process a batch of data.

        Args:
            batch (list[dict[str, torch.Tensor]]): A list of dictionaries containing tensors.

        Returns:
            DatasetBatch: The processed batch of data.

        Raises:
            NotImplementedError: This abstract method should be implemented in a subclass.
        """
        raise NotImplementedError
