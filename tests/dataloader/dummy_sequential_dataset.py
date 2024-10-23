from pydantic import BaseModel
from torch.utils.data.dataset import Dataset as TorchdataSet


class TestDataset(TorchdataSet):
    def __init__(self, num_samples: int):
        self.samples = list(range(num_samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class TestDatasetConfig(BaseModel):
    num_samples: int
