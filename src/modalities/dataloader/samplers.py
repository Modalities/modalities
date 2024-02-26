from torch.utils.data import BatchSampler, Sampler


class ResumableBatchSampler(Sampler):
    def __init__(self, start_index: int, underlying_batch_sampler: BatchSampler):
        """Sampler which starts at a specified batch index and continues sampling for
            for a given sampler. Works with normal samplers and BatchSamplers.

        Args:
            start_index (int): index to start sampling from
            existing_sampler (Sampler): Sampler from which we want to continue
        """

        self.start_index = start_index
        self.underlying_batch_sampler = underlying_batch_sampler
        # NOTE: we are only iterating ove the indices not the actual data
        # so this is relatively cheap
        self.indices = list(iter(self.underlying_batch_sampler))

    def __iter__(self):
        return iter(self.indices[self.start_index :])

    def __len__(self):
        return len(self.indices) - self.start_index

    @property
    def batch_size(self) -> int:
        return self.underlying_batch_sampler.batch_size
