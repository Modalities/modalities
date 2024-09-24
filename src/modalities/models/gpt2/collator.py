from abc import ABC, abstractmethod

import torch

from modalities.batch import DatasetBatch


class CollateFnIF(ABC):
    """CollateFnIF class to define a collate function interface."""

    @abstractmethod
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> DatasetBatch:
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


class GPT2LLMCollateFn(CollateFnIF):
    """GPT2LLMCollateFn class to define a collate function for GPT2 language model."""

    def __init__(self, sample_key: str, target_key: str):
        """
        Initializes the Collator object.

        Args:
            sample_key (str): The key for accessing the sample data.
            target_key (str): The key for accessing the target data.
        """
        self.sample_key = sample_key
        self.target_key = target_key

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> DatasetBatch:
        """
        Process a batch of data.

        Args:
            batch (list[dict[str, torch.Tensor]]): A list of dictionaries containing tensors.

        Returns:
            DatasetBatch: A processed batch of data where sample and target sequences are created.

        """

        sample_tensor = torch.stack([torch.tensor(d[self.sample_key]) for d in batch])
        samples = {self.sample_key: sample_tensor[:, :-1]}
        targets = {self.target_key: sample_tensor[:, 1:]}

        return DatasetBatch(targets=targets, samples=samples)
