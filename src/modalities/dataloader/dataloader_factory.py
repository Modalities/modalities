from typing import Callable, Optional

from torch.utils.data import BatchSampler
from torch.utils.data.dataset import Dataset

from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.samplers import ResumableBatchSampler


class DataloaderFactory:
    @staticmethod
    def get_dataloader(
        dataloader_tag: str,
        dataset: Dataset,
        batch_sampler: BatchSampler,
        collate_fn: Callable,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
        skip_num_steps: Optional[int] = 0,
    ) -> LLMDataLoader:
        batch_sampler = ResumableBatchSampler(start_index=skip_num_steps, underlying_batch_sampler=batch_sampler)

        dataloader = LLMDataLoader(
            dataloader_tag=dataloader_tag,
            batch_sampler=batch_sampler,
            dataset=dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

        return dataloader
