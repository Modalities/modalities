from torch.utils.data import Sampler


class ResumableSampler(Sampler):
    def __init__(self, start_index: int, existing_sampler: Sampler):
        """Sampler which starts at a specified index and continues sampling for
            for a given sampler. Works with normal samplers and BatchSamplers.

            NOTE: In its current implementation the entire index from `existing_sampler`
            is loaded into RAM.

        Args:
            start_index (int): index to start sampling from
            existing_sampler (Sampler): Sampler from which we want to continue
        """

        self.start_index = start_index
        self.existing_sampler = existing_sampler
        self.indices = list(iter(self.existing_sampler))

    def __iter__(self):
        return iter(self.indices[self.start_index :])

    def __len__(self):
        return len(self.indices) - self.start_index
