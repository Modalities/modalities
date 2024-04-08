import numpy as np

from modalities.dataloader.dataset import DummyDataset, DummySampleConfig, DummySampleDataType


def test_dummy_dataset():
    dataset = DummyDataset(
        num_samples=50,
        sample_definition=[
            DummySampleConfig(sample_key="input_ids", sample_shape=(512,), sample_type=DummySampleDataType.INT),
            DummySampleConfig(sample_key="images", sample_shape=(3, 224, 224), sample_type=DummySampleDataType.FLOAT),
        ],
    )
    assert len(dataset) == 50
    sample = next(iter(dataset))
    assert "input_ids" in sample
    assert sample["input_ids"].shape == (512,)
    assert sample["input_ids"].dtype == np.int64
    assert "images" in sample
    assert sample["images"].shape == (3, 224, 224)
    assert sample["images"].dtype == np.float64
