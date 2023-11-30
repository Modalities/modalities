import pytest

from llm_gym.dataloader.dataset import PackedMemMapDatasetContinuous, PackedMemMapDatasetMegatron


@pytest.mark.parametrize("block_size, expected_length", [(1, 4), (2, 3), (3, 3), (10, 2), (6, 2), (20, 1), (25, 0)])
def test_packed_megatron_dataset_loading(dummy_packed_data_path, block_size, expected_length):
    ds = PackedMemMapDatasetMegatron(dummy_packed_data_path, block_size)
    assert len(ds) == expected_length


@pytest.mark.parametrize("block_size, expected_length", [(1, 20), (2, 10), (3, 6), (10, 2), (6, 3), (20, 1), (25, 0)])
def test_packed_continuous_dataset_loading(dummy_packed_data_path, block_size, expected_length):
    ds = PackedMemMapDatasetContinuous(dummy_packed_data_path, block_size)
    assert len(ds) == expected_length
