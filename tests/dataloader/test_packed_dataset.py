import json
from pathlib import Path

import pytest

from modalities.dataloader.create_packed_data import EmbeddedStreamData, PackedDataGenerator, join_embedded_stream_data
from modalities.dataloader.dataset import PackedMemMapDatasetContinuous, PackedMemMapDatasetMegatron
from modalities.models.gpt2.collator import GPT2LLMCollateFn


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
        default_packed_dataset_path, block_size=block_size, sample_key="input_ids"
    )

    # read in the raw jsonl files for manual tokenization
    with open(indexed_dummy_data_path_long.raw_data_path) as f:
        jsonl_list = [json.loads(line)["text"] for line in f]

    jsonl_tokenized = [wrapped_gpt2_tokenizer.tokenize(v) for v in jsonl_list]
    eod_token_id = wrapped_gpt2_tokenizer.get_token_id("<|endoftext|>")
    # we flatten the list of tokenized documents and add the eod token at the end of each document
    jsonl_tokenized_flat = [token_id for doc in jsonl_tokenized for token_id in doc + [eod_token_id]]
    # we make sure that the length of the flattened tokenized jsonl file is a multiple of the block size
    # as the packed dataset also cuts off partially packed samples at the end.
    jsonl_tokenized_flat = jsonl_tokenized_flat[: len(jsonl_tokenized_flat) // block_size * block_size]

    # flatten the tokens from the packed dataset
    packed_dataset_tokens_flat = [j for i in iter(packed_dataset) for j in i["input_ids"].tolist()]

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
    assert [v for batch in loaded_dataset for v in batch["whatever"]] == [
        v for ds in original_datasets for batch in ds for v in batch["whatever"]
    ]


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
