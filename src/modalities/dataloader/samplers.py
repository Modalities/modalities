from typing import Optional

from torch.utils.data import BatchSampler, Sampler


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
