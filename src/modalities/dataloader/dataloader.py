from typing import Iterable, Optional, Union

from torch.utils.data import BatchSampler, Dataset, Sampler
from torch.utils.data.dataloader import DataLoader, T_co, _collate_fn_t, _worker_init_fn_t


class LLMDataLoader(DataLoader[T_co]):
    """LLMDataLoader is a custom DataLoader class that extends the PyTorch DataLoader class."""

    def __init__(
        self,
        dataloader_tag: str,
        batch_sampler: BatchSampler,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        sampler: Union[Sampler, Iterable, None] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        """
        Initializes a DataLoader object.

        Args:
            dataloader_tag (str): The tag for the dataloader.
            batch_sampler (BatchSampler): The batch sampler used for sampling batches.
            dataset (Dataset[T_co]): The dataset to load the data from.
            batch_size (Optional[int], optional): The number of samples per batch. Defaults to 1.
            sampler (Union[Sampler, Iterable, None], optional): The sampler used for sampling data. Defaults to None.
            num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 0.
            collate_fn (Optional[_collate_fn_t], optional): The function used to collate the data samples.
              Defaults to None.
            pin_memory (bool, optional): Flag indicating whether to pin the memory. Defaults to False.
            drop_last (bool, optional): Flag indicating whether to drop the last incomplete batch. Defaults to False.
            timeout (float, optional): The timeout value for collecting a batch from workers. Defaults to 0.
            worker_init_fn (Optional[_worker_init_fn_t], optional): The function used to initialize worker processes.
              Defaults to None.
            multiprocessing_context ([type], optional): The multiprocessing context to use. Defaults to None.
            generator ([type], optional): The random number generator. Defaults to None.
            prefetch_factor (Optional[int], optional): The number of batches to prefetch. Defaults to None.
            persistent_workers (bool, optional): Flag indicating whether to keep the workers alive
              between data loading iterations. Defaults to False.
            pin_memory_device (str, optional): The device to pin the memory to. Defaults to "".

        Returns:
            None
        """
        assert batch_sampler is not None and batch_size == 1
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # shuffling must be implemented on a dataset level
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

        self._dataloader_tag = dataloader_tag

    @property
    def dataloader_tag(self) -> str:
        """
        Returns the dataloader tag.

        Returns:
            str: The dataloader tag.
        """
        return self._dataloader_tag
