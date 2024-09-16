from typing import Callable, Optional

from torch.utils.data import BatchSampler
from torch.utils.data.dataset import Dataset

from modalities.dataloader.dataloader import LLMDataLoader, RepeatingDataLoader
from modalities.dataloader.samplers import ResumableBatchSampler
from modalities.exceptions import ConfigError


class DataloaderFactory:
    @staticmethod
    def get_dataloader(
        dataloader_tag: str,
        dataset: Dataset,
        batch_sampler: BatchSampler,
        collate_fn: Callable,
        num_workers: int,
        pin_memory: bool,
        skip_num_batches: Optional[int] = 0,
        fixed_num_batches: Optional[int] = None,
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
            skip_num_batches (int, optional): Defines the number of batches to skip.
              NOTE: The checkpoints are indexed with training steps (i.e., number of optimizer steps).
              skip_num_batches must not be confused with the number of optimizer steps!
              skip_num_batches = num optimizer steps * gradient accumulation steps
              Defaults to 0.
            fixed_num_batches: (int, optional): Fixed length of the dataloader by cutting off subsequent batches.
                Note that these are NOT the global number of batches, but the amount of batches that an
                individual rank sees. Make sure that the dataloader has at least fixed_num_batches.
                Defaults to None.

        Returns:
            LLMDataLoader: Instance of LLMDataLoader
        """

        batch_sampler = ResumableBatchSampler(
            start_index=skip_num_batches, underlying_batch_sampler=batch_sampler, max_num_elements=fixed_num_batches
        )

        if fixed_num_batches is not None and fixed_num_batches <= skip_num_batches:
            raise ConfigError("fixed_num_batches must be larger than skip_num_batches")

        # make sure that the batch sampler has enough elements such that we can fix the number of batches to num_batches
        if fixed_num_batches is not None and len(batch_sampler) < fixed_num_batches - skip_num_batches:
            raise ConfigError(
                f"The dataloader contains only {len(batch_sampler)} batches, which is less than "
                f"specified fixed amount of batches of {fixed_num_batches}."
            )

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
