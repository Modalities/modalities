from modalities.dataloader.dataloader import LLMDataLoader
import pytest

from modalities.dataloader.create_packed_data import PackedDataGenerator
from modalities.dataloader.dataset import PackedMemMapDatasetContinuous, PackedMemMapDatasetMegatron


@pytest.mark.parametrize("block_size, expected_length", [(1, 4), (2, 3), (3, 3), (10, 2), (6, 2), (20, 1), (25, 0)])
def test_packed_megatron_dataset_loading(dummy_packed_data_path, block_size, expected_length):
    ds = PackedMemMapDatasetMegatron(dummy_packed_data_path, block_size, sample_key="input_ids")
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
    ds = PackedMemMapDatasetContinuous(dummy_packed_data_path, block_size, sample_key="input_ids")
    assert len(ds) == expected_length
    dl = LLMDataLoader(dataloader_tag="unittest", dataset=ds)
    retrieved_input_ids = [list(x["input_ids"]) for x in dl]
    assert retrieved_input_ids == expected_output


def test_packed_continuous_dataset_missing_file(dummy_packed_data_path):
    dummy_packed_data_path.unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        PackedMemMapDatasetContinuous(dummy_packed_data_path, block_size=10, sample_key="input_ids")


@pytest.mark.parametrize("max_num_of_tokens, expected_index_size", [(None, 12), (10, 1)])
def test_create_packed_dataset(indexed_dummy_data_path, gpt2_tokenizer, max_num_of_tokens, expected_index_size):
    block_size = 5
    packed_generator = PackedDataGenerator(
        src_path=indexed_dummy_data_path.raw_data_path, tokenizer=gpt2_tokenizer, max_number_of_tokens=max_num_of_tokens
    )
    default_packed_dataset_path = packed_generator._default_destination_path()
    assert not default_packed_dataset_path.is_file()
    packed_generator.run()
    packed_dataset = PackedMemMapDatasetContinuous(
        default_packed_dataset_path, block_size=block_size, sample_key="input_ids"
    )

    start_of_jsonl_content = "0 Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor"
    tokenized_start_of_jsonl_content = gpt2_tokenizer(start_of_jsonl_content)["input_ids"]
    packed_dataset_iterator = iter(packed_dataset)
    assert tokenized_start_of_jsonl_content[:block_size] == next(packed_dataset_iterator)["input_ids"]
    assert tokenized_start_of_jsonl_content[block_size : 2 * block_size] == next(packed_dataset_iterator)["input_ids"]
    assert len(packed_dataset.index_base) == expected_index_size

    # check validity of index section in packed dataset
    for idx, (offset, entry_length) in enumerate(packed_dataset.index_base[:-1]):
        assert offset + entry_length == packed_dataset.index_base[idx + 1][0]
