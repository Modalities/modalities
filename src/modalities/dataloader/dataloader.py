import multiprocessing
from typing import Iterable, Optional

import webdataset as wd
from torch.utils.data import Dataset, DistributedSampler, Sampler
from torch.utils.data.dataloader import DataLoader, T_co, _collate_fn_t, _worker_init_fn_t

from modalities.dataloader.samplers import ResumableBatchSampler


class DataLoaderIF:
    pass


class LLMDataLoader(DataLoaderIF):
    """LLMDataLoader is a custom DataLoader class that extends the PyTorch DataLoader class."""

    def __init__(
        self,
        dataloader_tag: str,
        batch_sampler: ResumableBatchSampler,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        sampler: Sampler | Iterable | None = None,
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
            batch_sampler (ResumableBatchSampler): The batch sampler used for sampling batches.
            dataset (Dataset[T_co]): The dataset to load the data from.
            batch_size (Optional[int], optional): The number of samples per batch. Defaults to 1.
            sampler (Sampler | Iterable | None, optional): The sampler used for sampling data. Defaults to None.
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
        self._dataloader_tag = dataloader_tag
        self._batch_size = batch_sampler.batch_size
        self._torch_dataloader = DataLoader(
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

    @property
    def dataloader_tag(self) -> str:
        """
        Returns the dataloader tag.

        Returns:
            str: The dataloader tag.
        """
        return self._dataloader_tag

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size used in the dataloader.
        The batch size is the number of samples in each batch of data.

        Returns:
            int: The batch size used in the dataloader.

        Note:
            The parent Dataloader class has already a batch_size property defined which is originally used
            when the batch_sampler is not specified. Since the  LLMDataLoader enforces to always use a BatchSampler,
            we defined/ override the property batch_size to return the actual batch size used in the dataloder.
            BatchSampler is required, as we must seek forward in the dataloder during a warm start and
            we don't want to load all the data during the fast-forward.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        """
        Set the batch size for the dataloader.

        Parameters:
            value (int): The batch size to be set.

        Returns:
            None
        """
        self._batch_size = value

    def __len__(self):
        return self._torch_dataloader.__len__()

    def __iter__(self):
        return self._torch_dataloader.__iter__()

    @property
    def dataset(self) -> Dataset[T_co]:
        return self._torch_dataloader.dataset

    @property
    def batch_sampler(self) -> ResumableBatchSampler:
        return self._torch_dataloader.batch_sampler

    @property
    def sampler(self) -> Sampler | Iterable | None:
        return self._torch_dataloader.sampler

    @property
    def collate_fn(self) -> _collate_fn_t:
        return self._torch_dataloader.collate_fn

    @property
    def multiprocessing_context(self) -> str | multiprocessing.context.BaseContext:
        return self._torch_dataloader.multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        self._torch_dataloader.multiprocessing_context = multiprocessing_context

    @property
    def _auto_collation(self):
        return self._torch_dataloader._auto_collation

    @property
    def _index_sampler(self):
        return self._torch_dataloader._index_sampler

    def check_worker_number_rationality(self):
        return self._torch_dataloader.check_worker_number_rationality()

    @property
    def fast_forward_batch_id(self) -> int:
        """
        The batch ID until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward batch ID
        """
        return self._torch_dataloader.batch_sampler.start_index


class RepeatingDataLoader(LLMDataLoader):
    """
    RepeatingDataLoader is a custom DataLoader class that repeats the given dataloader
      for the specified number of epochs."""

    def __init__(self, dataloader: LLMDataLoader, num_epochs: int, reshuffle_after_epoch: bool = False):
        """
        Initializes a RepeatingDataLoader object that repeats the given dataloader for the specified number of epochs.
        This is especially useful for DataLoader types that we wish to automatically restart upon completion.

        Args:
            dataloader (LLMDataLoader[T_co]): The dataloader to be wrapped.
            num_epochs (int): The number of epochs to iterate over the dataloader.
            reshuffle_after_epoch (bool, optional): Flag indicating whether to reshuffle the dataloader
              after each epoch. Defaults to False.

        Returns:
            None

        Note:
            Based on: https://github.com/microsoft/DeepSpeed/blob/99951caa3d2155a3bb84109a0828543793e088cc/deepspeed/runtime/dataloader.py#L17
        """
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        self.current_epoch = 0
        self.reshuffle_after_epoch = reshuffle_after_epoch
        self.num_epochs = num_epochs

    def __iter__(self):
        """
        Returns an iterator object for the DataLoader.
        """
        return self

    def __next__(self):
        """
        Returns the next batch of data from the DataLoader.

        Raises:
            StopIteration: If there are no more batches of data to return.

        Returns:
            batch: The next batch of data.
        """
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
        """
        Returns the dataloader tag.

        Returns:
            str: The dataloader tag.
        """
        return self.dataloader.dataloader_tag

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size used by the dataloader.

        Returns:
            int: The batch size used by the dataloader.
        """
        return self.dataloader.batch_size

    @property
    def fast_forward_batch_id(self) -> int:
        """
        The batch ID until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward batch id
        """
        return self.dataloader.fast_forward_batch_id

    def __len__(self) -> int:
        """
        Returns the total number of steps in the dataloader.

        Returns:
            int: The total number of steps.
        """
        return self.num_epochs * len(self.dataloader)


class WebDataLoader(DataLoaderIF):
    def __init__(
        self,
        dataloader_tag: str,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        self.num_batches = len(dataset) // batch_size + int(not drop_last)
        dataset = dataset.batched(batch_size, collation_fn=collate_fn)
        self.webloader = wd.WebLoader(dataset=dataset, batch_size=None, num_workers=num_workers, pin_memory=pin_memory)
        self.webloader = self.webloader.with_epoch(self.num_batches)
        self.dataloader_tag = dataloader_tag
        self.batch_size = batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return iter(self.webloader)

    @property
    def batch_size(self) -> int:
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
        return 0  # self.batch_size * self.batch_sampler.start_index

    @property
    def fast_forward_batch_id(self) -> int:
        """The batch id until which we fast-forward, as specified in the ResumableBatchSampler.

        Returns:
            int: fast forward batch id
        """
        return 0  # self.batch_sampler.start_index
