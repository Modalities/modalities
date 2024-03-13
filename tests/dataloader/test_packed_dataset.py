from pathlib import Path

import numpy as np
import pytest

from modalities.dataloader.create_packed_data import EmbeddedStreamData, PackedDataGenerator, join_embedded_stream_data
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
    retrieved_input_ids = [list(packed_samples["input_ids"]) for packed_samples in ds]
    assert retrieved_input_ids == expected_output


def test_packed_continuous_dataset_missing_file(dummy_packed_data_path):
    dummy_packed_data_path.unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        PackedMemMapDatasetContinuous(dummy_packed_data_path, block_size=10, sample_key="input_ids")


def test_create_packed_dataset(indexed_dummy_data_path, gpt2_tokenizer):
    block_size = 5
    packed_generator = PackedDataGenerator(
        src_path=indexed_dummy_data_path.raw_data_path, tokenizer=gpt2_tokenizer, number_of_processes=2
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
    np.testing.assert_equal(tokenized_start_of_jsonl_content[:block_size], next(packed_dataset_iterator)["input_ids"])
    np.testing.assert_equal(
        tokenized_start_of_jsonl_content[block_size : 2 * block_size], next(packed_dataset_iterator)["input_ids"]
    )
    assert len(packed_dataset._embedded_stream_data.index_base) == 12

    # check validity of index section in packed dataset
    for idx, (offset, entry_length) in enumerate(packed_dataset._embedded_stream_data.index_base[:-1]):
        assert offset + entry_length == packed_dataset._embedded_stream_data.index_base[idx + 1][0]


def test_join_packed_datasets(dummy_packed_data_path, tmpdir):
    packed_data_clones = [Path(tmpdir, f"clone{i}.pbin") for i in range(3)]
    for clone in packed_data_clones:
        clone.write_bytes(dummy_packed_data_path.read_bytes())

    joined_target_file = Path(tmpdir, "joined.pbin")

    stream_data = list(map(EmbeddedStreamData, packed_data_clones))
    join_embedded_stream_data(stream_data, joined_target_file)

    loaded_joint_data = EmbeddedStreamData(joined_target_file)
    assert loaded_joint_data
    assert loaded_joint_data.data_len == sum(d.data_len for d in stream_data)

    loaded_dataset = PackedMemMapDatasetContinuous(joined_target_file, block_size=2, sample_key="whatever")
    original_datasets = [
        PackedMemMapDatasetContinuous(p, block_size=2, sample_key="whatever") for p in packed_data_clones
    ]
    assert [v for batch in loaded_dataset for v in batch["whatever"]] == [
        v for ds in original_datasets for batch in ds for v in batch["whatever"]
    ]
