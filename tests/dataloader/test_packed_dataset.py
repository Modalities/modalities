import json
from pathlib import Path
from typing import Type

import numpy as np
import pytest

from modalities.dataloader.create_packed_data import EmbeddedStreamData, PackedDataGenerator, join_embedded_stream_data
from modalities.dataloader.dataset import (
    PackedMemMapDatasetBase,
    PackedMemMapDatasetContinuous,
    PackedMemMapDatasetMegatron,
)
from modalities.models.gpt2.collator import GPT2LLMCollateFn


@pytest.mark.parametrize("block_size, expected_length", [(1, 4), (2, 3), (3, 3), (10, 2), (6, 2), (20, 1), (25, 0)])
def test_packed_megatron_dataset_loading(dummy_packed_data_path, block_size, expected_length):
    ds = PackedMemMapDatasetMegatron(
        raw_data_path=dummy_packed_data_path, block_size=block_size, sample_key="input_ids"
    )
    assert len(ds) == expected_length


@pytest.mark.parametrize(
    "block_size, expected_length, expected_output",
    [
        (
            2,
            19,
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [12, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [16, 17],
                [17, 18],
                [18, 19],
            ],
        ),
        (
            3,
            9,
            [
                [0, 1, 2],
                [2, 3, 4],
                [4, 5, 6],
                [6, 7, 8],
                [8, 9, 10],
                [10, 11, 12],
                [12, 13, 14],
                [14, 15, 16],
                [16, 17, 18],
            ],
        ),
        (10, 2, [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]),
        (6, 3, [[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10], [10, 11, 12, 13, 14, 15]]),
        (20, 1, [list(range(20))]),
        (21, 0, ValueError),
        (1, 0, ValueError),
    ],
)
def test_packed_continuous_dataset_loading(dummy_packed_data_path, block_size, expected_length, expected_output):
    try:
        ds = PackedMemMapDatasetContinuous(
            raw_data_path=dummy_packed_data_path, block_size=block_size, sample_key="input_ids"
        )
    except ValueError:
        assert expected_output == ValueError
        return

    assert len(ds) == expected_length
    retrieved_input_ids = [list(packed_samples["input_ids"]) for packed_samples in ds]
    assert retrieved_input_ids == expected_output


def test_packed_continuous_dataset_missing_file(dummy_packed_data_path):
    dummy_packed_data_path.unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        PackedMemMapDatasetContinuous(dummy_packed_data_path, block_size=10, sample_key="input_ids")


def test_create_packed_dataset(indexed_dummy_data_path_long, wrapped_gpt2_tokenizer):
    # In this test, we create a packed dataset from a long jsonl file
    # and iterate over the packed dataset to check if the tokenization is correct.
    # We do so by manually tokenizing the jsonl file and comparing the tokenized
    # output with the packed dataset
    block_size = 20
    packed_generator = PackedDataGenerator(
        src_path=indexed_dummy_data_path_long.raw_data_path,
        tokenizer=wrapped_gpt2_tokenizer,
        number_of_processes=5,
        eod_token="<|endoftext|>",
        index_path=indexed_dummy_data_path_long.index_path,
        jq_pattern=".text",
        processing_batch_size=5,
        raw_samples_queue_size=3,
        processed_samples_queue_size=3,
    )
    default_packed_dataset_path = packed_generator._default_destination_path()
    assert not default_packed_dataset_path.is_file()
    packed_generator.run()
    packed_dataset = PackedMemMapDatasetContinuous(
        default_packed_dataset_path, block_size=block_size, sample_key="input_ids", load_index=True
    )

    # read in the raw jsonl files for manual tokenization
    with open(indexed_dummy_data_path_long.raw_data_path) as f:
        jsonl_list = [json.loads(line)["text"] for line in f]

    jsonl_tokenized = [wrapped_gpt2_tokenizer.tokenize(v) for v in jsonl_list]
    eod_token_id = wrapped_gpt2_tokenizer.get_token_id("<|endoftext|>")
    # we flatten the list of tokenized documents and add the eod token at the end of each document
    jsonl_tokenized_flat = [token_id for doc in jsonl_tokenized for token_id in doc + [eod_token_id]]

    # we calculate the number of samples in the jsonl file given the block size
    # the formula takes into account that that from the second sample onwards the
    # last token (i.e., last target token) is reused as the first input token from the next sample
    num_samples = (len(jsonl_tokenized_flat) - block_size) // (block_size - 1) + 1
    # the first sample has a length of block_size and the subsequent one of block_size-1
    num_tokens = block_size + (block_size - 1) * (num_samples - 1)
    jsonl_tokenized_flat = jsonl_tokenized_flat[:num_tokens]

    # flatten the tokens from the packed dataset to reproduce the tokenized jsonl file
    packed_dataset_tokens_flat = []
    for block_id, block in enumerate(iter(packed_dataset)):
        if block_id > 0:
            # we remove the first token from each block as it is a
            # reused token from the previous block
            tokens = block["input_ids"].tolist()[1:]
            packed_dataset_tokens_flat += tokens
        else:
            packed_dataset_tokens_flat += block["input_ids"].tolist()

    # compare the flattened tokens from the packed dataset with the manually tokenized jsonl file
    assert packed_dataset_tokens_flat == jsonl_tokenized_flat

    # make sure that each packed sample in the packed dataset has a length of block_size
    for sample in iter(packed_dataset):
        assert len(sample["input_ids"]) == block_size

    assert len(packed_dataset._embedded_stream_data.index_base) == 500

    # check validity of index section in packed dataset
    # we make sure that the offset is calculated correctly based on the length of the entry and the previous index
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

    original_datasets_concatenated = []
    for ds_id, ds in enumerate(original_datasets):
        for batch_id, batch in enumerate(ds):
            if ds_id > 0 and batch_id == 0:
                # we add the batch that was missing from the transition from one dataset to the next
                # NOTE: this test only works with block_size=2!
                original_datasets_concatenated += [
                    original_datasets_concatenated[-1],
                    batch["whatever"].flatten().tolist()[0],
                ]

            original_datasets_concatenated += batch["whatever"].flatten().tolist()

            print([batch["whatever"].tolist()])
    loaded_dataset_flattened = [v for batch in loaded_dataset for v in batch["whatever"]]
    assert loaded_dataset_flattened == original_datasets_concatenated


@pytest.mark.parametrize("token_size_in_bytes", [1, 2, 4])
def test_conversion_tokens_represented_as_unsigned_ints(tmpdir, token_size_in_bytes: int):
    src_pbin_path = Path(__file__).parents[2] / "data/lorem_ipsum.pbin"
    pbin_path = Path(tmpdir, "lorem_ipsum.pbin")
    pbin_path.write_bytes(src_pbin_path.read_bytes())
    with pbin_path.open("r+b") as fin:
        fin.seek(8)
        fin.write(token_size_in_bytes.to_bytes(4, byteorder="little"))
    assert pbin_path.is_file()
    sample_key = "input_ids"
    ds = PackedMemMapDatasetContinuous(raw_data_path=pbin_path, block_size=10, sample_key=sample_key)
    assert list(ds)

    collator = GPT2LLMCollateFn(sample_key=sample_key, target_key="abc")
    for batch in zip(ds, ds):
        collator(list(batch))


@pytest.fixture
def packed_dataset(indexed_dummy_data_path_long, wrapped_gpt2_tokenizer) -> PackedMemMapDatasetBase:
    packed_generator = PackedDataGenerator(
        src_path=indexed_dummy_data_path_long.raw_data_path,
        tokenizer=wrapped_gpt2_tokenizer,
        number_of_processes=5,
        eod_token="<|endoftext|>",
        index_path=indexed_dummy_data_path_long.index_path,
        jq_pattern=".text",
        processing_batch_size=5,
        raw_samples_queue_size=3,
        processed_samples_queue_size=3,
    )
    default_packed_dataset_path = packed_generator._default_destination_path()
    assert not default_packed_dataset_path.is_file()
    packed_generator.run()
    dataset = PackedMemMapDatasetBase(default_packed_dataset_path, sample_key="input_ids")
    return dataset


@pytest.fixture
def tokenized_jsonl_data(indexed_dummy_data_path_long, wrapped_gpt2_tokenizer) -> list[list[int]]:
    # read in the raw jsonl files for manual tokenization
    with open(indexed_dummy_data_path_long.raw_data_path, "r", encoding="utf-8") as f:
        jsonl_list = [json.loads(line)["text"] for line in f]

    eod_token_id = wrapped_gpt2_tokenizer.get_token_id("<|endoftext|>")
    jsonl_tokenized = [wrapped_gpt2_tokenizer.tokenize(v) + [eod_token_id] for v in jsonl_list]
    return jsonl_tokenized


def test_original_samples_in_packed_dataset(
    packed_dataset: PackedMemMapDatasetBase, tokenized_jsonl_data: list[list[int]]
):
    # In this test, we create a packed dataset from a long jsonl file
    # and iterate over the packed dataset to check if the tokenization is correct.
    # We do so by manually tokenizing the jsonl file and comparing the tokenized
    # output with the packed dataset

    for sample, original_sample in zip(packed_dataset, tokenized_jsonl_data):
        assert sample["input_ids"].tolist() == original_sample


@pytest.mark.parametrize(
    "slice, expected_error",
    [
        ((0, 10), None),
        ((0, 100), None),
        ((0, 500), None),
        ((0, 501), IndexError),
        ((5, 10), None),
        ((5, 100), None),
        ((5, 500), None),
        ((5, 501), IndexError),
        ((5, -1), None),
        ((-3, -1), None),
        ((3, 1), ValueError),
        ((3, None), None),
        ((None, None), None),
        ((500, 501), IndexError),
        ((450, 450), None),
    ],
)
def test_original_samples_in_packed_dataset_slicing(
    packed_dataset: PackedMemMapDatasetBase,
    tokenized_jsonl_data: list[list[int]],
    slice: tuple[int, int],
    expected_error: Type[Exception],
):
    if expected_error is not None:
        with pytest.raises(expected_error):
            packed_dataset[slice[0] : slice[1]]
        return

    for sample, original_sample in zip(
        packed_dataset[slice[0] : slice[1]]["input_ids"], tokenized_jsonl_data[slice[0] : slice[1]]
    ):
        assert sample.tolist() == original_sample


@pytest.mark.parametrize(
    "token_size_in_bytes, block_size, total_tokens", [(1, 32, 32), (2, 32, 512), (4, 32, 1000), (4, 32, 1234)]
)
def test_continuously_packed_index(token_size_in_bytes: int, block_size: int, total_tokens: int):
    num_samples = (total_tokens - block_size) // (block_size - 1) + 1
    # given num_samples we calculate the starting index and length of each sample as tuple.
    result_slow = [
        ((i * block_size - i) * token_size_in_bytes, block_size * token_size_in_bytes) for i in range(num_samples)
    ]

    result_vectorized = PackedMemMapDatasetContinuous._create_packed_index(
        total_tokens=total_tokens, block_size=block_size, token_size_in_bytes=token_size_in_bytes
    )

    assert np.all(result_slow == result_vectorized)


@pytest.mark.parametrize(
    "vocab_size, expected_num_bytes",
    [(254, 1), (255, 1), (256, 1), (257, 2), (65534, 2), (65535, 2), (65536, 2), (65537, 4), (65538, 4), (10000000, 4)],
)
def test__get_required_num_of_bytes_to_repr(vocab_size: int, expected_num_bytes: int):
    num_bytes = PackedDataGenerator._get_required_num_of_bytes_to_repr(int_to_get_repr=vocab_size)
    assert expected_num_bytes == num_bytes
