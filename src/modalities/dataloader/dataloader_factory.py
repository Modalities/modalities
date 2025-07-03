from torch.utils.data import BatchSampler
from torch.utils.data.dataset import Dataset

from modalities.dataloader.collate_fns.collate_if import CollatorIF
from modalities.dataloader.dataloader import LLMDataLoader


class DataloaderFactory:
    @staticmethod
    def get_dataloader(
        dataloader_tag: str,
        dataset: Dataset,
        batch_sampler: BatchSampler,
        collator: CollatorIF,
        num_workers: int,
        pin_memory: bool,
    ) -> LLMDataLoader:
        """
        Factory method for the instantiation of LLMDataLoader.

        Args:
            dataloader_tag (str): Tag for the dataloader
            dataset (Dataset): Dataset to be used
            batch_sampler (BatchSampler): batch sampler for batch-wise sampling from the dataset
            collator (CollatorIF): Collator for shaping the batch
            num_workers (int): Number of workers for the dataloader
            pin_memory (bool): Flag indicating whether to pin memory
        Returns:
            LLMDataLoader: Instance of LLMDataLoader
        """
        dataloader = LLMDataLoader(
            dataloader_tag=dataloader_tag,
            batch_sampler=batch_sampler,
            dataset=dataset,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataloader
