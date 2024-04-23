from typing import Iterable, Optional, Union

from torch.utils.data import Dataset, DistributedSampler, Sampler
from torch.utils.data.dataloader import DataLoader, T_co, _collate_fn_t, _worker_init_fn_t

from modalities.dataloader.samplers import ResumableBatchSampler


class LLMDataLoader(DataLoader[T_co]):
    def __init__(
        self,
        dataloader_tag: str,
        batch_sampler: ResumableBatchSampler,
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
    def fast_forward_batch_id(self) -> int:
        """The batch id until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward batch id
        """
        return self.batch_sampler.start_index


class RepeatingDataLoader(LLMDataLoader[T_co]):
    def __init__(self, dataloader: LLMDataLoader[T_co], num_epochs: int, reshuffle_after_epoch: bool = False):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Based on: https://github.com/microsoft/DeepSpeed/blob/99951caa3d2155a3bb84109a0828543793e088cc/deepspeed/runtime/dataloader.py#L17
        """
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        self.current_epoch = 0
        self.reshuffle_after_epoch = reshuffle_after_epoch
        self.num_epochs = num_epochs

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration as e:
            if self.dataloader.sampler is not None:
                self.current_epoch += 1
                # After finishing an epoch, we set the start_index to 0 to start from the beginning
                # The start_index might have been >0 in case of a warm start
                self.dataloader.batch_sampler.start_index = 0

                if self.reshuffle_after_epoch:
                    # In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating
                    # the DataLoader iterator is necessary to make shuffling work properly across multiple epochs.
                    # Otherwise, the same ordering will be always used. See discussion:
                    # https://discuss.pytorch.org/t/why-is-sampler-set-epoch-epoch-needed-for-distributedsampler/149672
                    if isinstance(self.dataloader.sampler, DistributedSampler):
                        self.dataloader.sampler.set_epoch(self.current_epoch)
                    else:
                        raise NotImplementedError(
                            "Reshuffling after each epoch is only supported for DistributedSampler"
                        )
            if self.current_epoch < self.num_epochs:
                self.data_iter = iter(self.dataloader)
                batch = next(self.data_iter)
            else:
                raise StopIteration(f"RepeatingDataLoader has completed after {self.current_epoch} epochs") from e
        return batch

    @property
    def dataloader_tag(self) -> str:
        return self.dataloader.dataloader_tag

    @property
    def batch_size(self) -> int:
        return self.dataloader.batch_size

    @property
    def fast_forward_batch_id(self) -> int:
        """The batch id until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward batch id
        """
        return self.dataloader.fast_forward_batch_id

    def __len__(self) -> int:
        return self.num_epochs * len(self.dataloader)
