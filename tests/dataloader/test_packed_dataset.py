import pytest

from llm_gym.dataloader.dataset import PackedMemMapDatasetContinuous, PackedMemMapDatasetMegatron
from llm_gym.dataset_loader import LLMDataLoader


@pytest.mark.parametrize("block_size, expected_length", [(1, 4), (2, 3), (3, 3), (10, 2), (6, 2), (20, 1), (25, 0)])
def test_packed_megatron_dataset_loading(dummy_packed_data_path, block_size, expected_length):
    ds = PackedMemMapDatasetMegatron(dummy_packed_data_path, block_size)
    assert len(ds) == expected_length


@pytest.mark.parametrize(
    "block_size, expected_length, expected_output",
    [
        (1, 20, [[i] for i in range(20)]),
        (2, 10, [[2 * i, 2 * i + 1] for i in range(10)]),
        (3, 6, [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(6)]),
        (10, 2, [list(range(10)), list(range(10, 20))]),
        (6, 3, [list(range(i * 6, i * 6 + 6)) for i in range(3)]),
        (20, 1, [list(range(20))]),
        (25, 0, []),
    ],
)
def test_packed_continuous_dataset_loading(dummy_packed_data_path, block_size, expected_length, expected_output):
    ds = PackedMemMapDatasetContinuous(dummy_packed_data_path, block_size)
    assert len(ds) == expected_length
    dl = LLMDataLoader(dataloader_tag="unittest", dataset=ds)
    retrieved_input_ids = [list(x["input_ids"]) for x in dl]
    assert retrieved_input_ids == expected_output


def test_packed_continuous_dataset_missing_file(dummy_packed_data_path):
    dummy_packed_data_path.unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        PackedMemMapDatasetContinuous(dummy_packed_data_path, block_size=10)
