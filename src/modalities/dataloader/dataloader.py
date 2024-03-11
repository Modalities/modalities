from typing import Iterable, Optional, Union

from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader, T_co, _collate_fn_t, _worker_init_fn_t
from torch.utils.data.sampler import BatchSampler


class LLMDataLoader(DataLoader[T_co]):
    def __init__(
        self,
        dataloader_tag: str,
        batch_sampler: BatchSampler,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
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
        assert batch_sampler is not None and batch_size == 1
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
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
        self._batch_size = batch_sampler.batch_size

    @property
    def dataloader_tag(self) -> str:
        return self._dataloader_tag

    @property
    def batch_size(self) -> int:
        # The parent Dataloader class has already a batch_size property defined which is originally used
        # when the batch_sampler is not specified. Since the  LLMDataLoader enforces to always use a BatchSampler,
        # we defined/ override the property batch_size to return the actual batch size used in the dataloder.
        # BatchSampler is required, as we must seek forward in the dataloder during a warm start and
        # we don't want to load all the data during the fast-forward.
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    @property
    def fast_forward_sample_id(self) -> int:
        """The sample id until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward sample id
        """
        return self.batch_size * self.batch_sampler.start_index

    @property
    def fast_forward_batch_id(self) -> int:
        """The batch id until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward batch id
        """
        return self.batch_sampler.start_index


class RepeatingDataLoader(LLMDataLoader[T_co]):
    def __init__(self, dataloader: LLMDataLoader[T_co], reshuffle_after_epoch: bool = False):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        self.current_epoch = 0
        self.reshuffle_after_epoch = reshuffle_after_epoch

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            if self.dataloader.sampler is not None:
                # In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating
                # the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
                # Otherwise, the same ordering will be always used. See discussion:
                # https://discuss.pytorch.org/t/why-is-sampler-set-epoch-epoch-needed-for-distributedsampler/149672
                self.current_epoch += 1
                self.dataloader.sampler.set_epoch(self.current_epoch)
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch

    @property
    def dataloader_tag(self) -> str:
        return self.dataloader._dataloader_tag

    @property
    def batch_size(self) -> int:
        return self.dataloader.batch_sampler.batch_size

    @property
    def fast_forward_sample_id(self) -> int:
        """The sample id until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward sample id
        """
        return self.dataloader.batch_size * self.batch_sampler.start_index

    @property
    def fast_forward_batch_id(self) -> int:
        """The batch id until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward batch id
        """
        return self.dataloader.batch_sampler.start_index
