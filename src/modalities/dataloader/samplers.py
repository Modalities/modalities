import math
from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Dataset, Sampler


class ResumableBatchSampler(Sampler):
    def __init__(
        self, start_index: int, underlying_batch_sampler: BatchSampler, max_num_elements: Optional[int] = None
    ):
        """
        Sampler which starts at a specified batch index and continues sampling for
            for a given sampler. Works with normal samplers and BatchSamplers.

        Args:
            start_index (int): index to start sampling from
            underlying_batch_sampler (BatchSampler): Sampler providing the batch ids.
            max_num_elements (Optional[int]): The maximum number of elements the sampler returns. Default None.

        Returns:
            None
        """

        self.start_index = start_index
        self.max_num_elements = max_num_elements
        self.underlying_batch_sampler = underlying_batch_sampler
        # NOTE: we are only iterating ove the indices not the actual data
        # so this is relatively cheap
        self.indices = list(iter(self.underlying_batch_sampler))
        # We discard the samples that come after max_num_elements
        # NOTE, that skipping is implemented in __iter__ and __len__.
        if self.max_num_elements is not None:
            self.indices = self.indices[:max_num_elements]

    def __iter__(self):
        """
        Returns an iterator over the indices starting from the start_index.

        Returns:
            iterator: An iterator over the indices.
        """
        return iter(self.indices[self.start_index :])

    def __len__(self):
        """
        Returns the length of the sampler, which is the number of indices minus the start index.

        Returns:
            int: The length of the sampler.
        """
        return len(self.indices) - self.start_index

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size of the underlying batch sampler.

        Returns:
            int: The batch size of the underlying batch sampler.
        """
        return self.underlying_batch_sampler.batch_size


T_co = TypeVar("T_co", covariant=True)


class ResumableDistributedSampler(Sampler[T_co]):
    """Sampler that restricts data loading to a subset of the dataset.
    We adopted this class from pytorch's DistributedSampler class and added the ability to resume from a specific index.
    source: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.
    """

    def __init__(
        self,
        dataset: Dataset,
        rank: int,
        num_replicas: Optional[int] = None,
        epoch: Optional[int] = 0,
        shuffle: Optional[bool] = False,
        seed: Optional[int] = 0,
        drop_last: Optional[bool] = False,
        skip_num_global_samples: Optional[int] = 0,
    ) -> None:
        """Instantiates a distributed and resumable Sampler object.

        Args:
            dataset (Dataset): The dataset to sample from.
            rank (int): The global rank of the current process.
            num_replicas (int, optional): Number of replicas.
                This usually equals the world size. Defaults to None.
            epoch (int, optional): Current epoch. Defaults to 0.
            shuffle (bool, optional): Boolean flag whether to shuffle the data. Defaults to False.
            seed (int, optional): Seed for the shuffling. Defaults to 0.
            drop_last (bool, optional): Boolean flag indicating whether to drop the last samples
                that cannot be distributed over all ranks (i.e., maximum world size - samples).
                If drop_last is false padding is applied for these samples, by resampling the initial samples.
                Defaults to False.
            skip_num_global_samples (int, optional): Number of samples to skip, e.g., due to warmstart.
                Defaults to 0.

        Raises:
            RuntimeError: Requires distributed package to be available if num_replicas is None.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        self.rank = rank
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.epoch = epoch
        self.drop_last = drop_last
        self.skip_num_global_samples = skip_num_global_samples

        self.global_num_samples = len(self.dataset) - self.skip_num_global_samples
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.global_num_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.local_num_samples = math.ceil(
                (self.global_num_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            # if this is not integer divisible, we will add padding by reusing the beginning of the data
            self.local_num_samples = math.ceil(self.global_num_samples / self.num_replicas)  # type: ignore[arg-type]

        # the actual number of samples we will be iterating over
        self.global_num_samples_effective = self.local_num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices_full = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices_full = list(range(len(self.dataset)))  # type: ignore[arg-type]

        indices_without_skipped = indices_full[self.skip_num_global_samples :]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.global_num_samples_effective - len(indices_without_skipped)
            if padding_size <= len(indices_full):
                indices_without_skipped += indices_full[:padding_size]  # we want to reuse the beginning of the data
            else:
                # if the padding size is larger than the data, we create an extended index by repeating the indices
                indices_without_skipped += (indices_full * math.ceil(padding_size / len(indices_full)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices_without_skipped = indices_without_skipped[: self.global_num_samples_effective]

        if len(indices_without_skipped) != self.global_num_samples_effective:
            raise ValueError(
                f"global_num_samples_effective ({self.global_num_samples_effective}) does not match the actual"
                f"number of samples ({len(indices_without_skipped)})"
            )

        # subsample
        indices_without_skipped = indices_without_skipped[
            self.rank : self.global_num_samples_effective : self.num_replicas
        ]

        if len(indices_without_skipped) != self.local_num_samples:
            raise ValueError(
                f"local_num_samples ({self.local_num_samples}) does not match the actual"
                f"number of samples ({len(indices_without_skipped)})"
            )

        return iter(indices_without_skipped)

    def __len__(self) -> int:
        return self.local_num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
