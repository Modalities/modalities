import torch
from pydantic import BaseModel

from modalities.dataloader.collate_fns.collate_if import CollateFnIF


class AutoregressiveCollateFnConfig(BaseModel):
    sample_key: str
    target_key: str


class AutoregressiveCollateFn(CollateFnIF):
    """AutoregressiveCollateFn class to define a collate function for language modeling."""

    def __init__(self, sample_key: str, target_key: str):
        """
        Initializes the Collator object.

        Args:
            sample_key (str): The key for accessing the sample data.
            target_key (str): The key for accessing the target data.
        """
        self.sample_key = sample_key
        self.target_key = target_key

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process a batch of data.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing tensors of the batch.

        Returns:
            dict[str, torch.Tensor]: The processed batch with sample and target tensors.
        """
        sample_tensor = batch[self.sample_key]
        batch[self.sample_key] = sample_tensor[:, :-1]
        batch[self.target_key] = sample_tensor[:, 1:]
        return batch
