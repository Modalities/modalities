from typing import Callable

from torch.utils.data import BatchSampler
from torch.utils.data.dataset import Dataset

from modalities.dataloader.dataloader import LLMDataLoader, RepeatingDataLoader


class DataloaderFactory:
    @staticmethod
    def get_dataloader(
        dataloader_tag: str,
        dataset: Dataset,
        batch_sampler: BatchSampler,
        collate_fn: Callable,
        num_workers: int,
        pin_memory: bool,
    ) -> LLMDataLoader:
        """
        Factory method for the instantiation of LLMDataLoader.

        Args:
            dataloader_tag (str): Tag for the dataloader
            dataset (Dataset): Dataset to be used
            batch_sampler (BatchSampler): batch sampler for batch-wise sampling from the dataset
            collate_fn (Callable): Callable for shaping the batch
            num_workers (int): Number of workers for the dataloader
            pin_memory (bool): Flag indicating whether to pin memory
        Returns:
            LLMDataLoader: Instance of LLMDataLoader
        """
        dataloader = LLMDataLoader(
            dataloader_tag=dataloader_tag,
            batch_sampler=batch_sampler,
            dataset=dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataloader

    @staticmethod
    def get_repeating_dataloader(
        dataloader: LLMDataLoader, num_epochs: int, reshuffle_after_epoch: bool = False
    ) -> RepeatingDataLoader:
        """
        Returns a RepeatingDataLoader object that repeats the given dataloader
          for the specified number of epochs.

        Parameters:
            dataloader (LLMDataLoader): The dataloader to be repeated.
            num_epochs (int): The number of times the dataloader should be repeated.
            reshuffle_after_epoch (bool, optional): Flag indicating whether to reshuffle
              the data after each epoch. Defaults to False.

        Returns:
            RepeatingDataLoader: A RepeatingDataLoader object that repeats the given dataloader
              for the specified number of epochs.
        """
        dataloader = RepeatingDataLoader(dataloader, num_epochs, reshuffle_after_epoch)
        return dataloader
