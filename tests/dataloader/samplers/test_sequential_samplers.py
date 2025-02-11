import pytest
from torch.utils.data import Dataset, SequentialSampler


class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.data = list(range(num_samples))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


@pytest.mark.parametrize(
    "num_samples, world_size",
    [
        (10, 3),
        (15, 4),
    ],
)
def test_distributed_setting(num_samples, world_size):
    dataset = DummyDataset(num_samples)
    samplers = [SequentialSampler(dataset) for _ in range(world_size)]

    expected_indices = list(range(num_samples))
    # Ensures that all ranks receive the exact same samples in the same order
    assert all(list(sampler) == expected_indices for sampler in samplers)
