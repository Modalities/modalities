from torch.utils.data import SequentialSampler, Dataset

class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.data = list(range(num_samples))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def test_distributed_setting():
    num_samples = 10
    dataset = DummyDataset(num_samples)
    world_size = 3
    samplers = [SequentialSampler(dataset)for _ in range(world_size)]
    
    expected_indices = list(range(num_samples))
    # Ensures that all ranks receive the exact same samples in the same order
    assert all(list(sampler) == expected_indices for sampler in samplers)
