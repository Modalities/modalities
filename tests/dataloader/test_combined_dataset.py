import pytest

from modalities.dataloader.dataset import CombinedDataset


@pytest.fixture
def dummy_dataset_1() -> list[int]:
    return list(range(10))


@pytest.fixture
def dummy_dataset_2() -> list[int]:
    return list(range(10, 15))


def test_combined_dataset(dummy_dataset_1: list[int], dummy_dataset_2: list[int]):
    combined_dataset = CombinedDataset(datasets=[dummy_dataset_1, dummy_dataset_2])

    # check that length is calculated correctly
    assert len(combined_dataset) == 15

    # check that the elements are iterated over in order
    assert [i for i in combined_dataset] == list(range(15))

    # check that we throw an error when trying to access an index that is out of bounds
    with pytest.raises(IndexError):
        combined_dataset[15]
