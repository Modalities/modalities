import json
import pytest

from PIL import Image

import numpy as np
import numpy.testing
from modalities.dataloader.codecs import HfTokenizerCodec, PillowImageCodec, TorchaudioAudioCodec
from modalities.dataloader.create_packed_data import PackedDataGenerator
from modalities.dataloader.dataset import PackedMemMapDataset, PackedMemMapDatasetContinuous, PackedMemMapDatasetMegatron


@pytest.mark.skip(reason="New packed data format not implemented for megatron dataset")
@pytest.mark.parametrize("block_size, expected_length", [(1, 4), (2, 3), (3, 3), (10, 2), (6, 2), (20, 1), (25, 0)])
def test_packed_megatron_dataset_loading(dummy_packed_data_path, block_size, expected_length):
    ds = PackedMemMapDatasetMegatron(dummy_packed_data_path, block_size, sample_key="input_ids")
    assert len(ds) == expected_length


def test_packed_dataset_loading(dummy_packed_data_path):
    
    ds = PackedMemMapDataset(
        dummy_packed_data_path,
        sample_keys=["input_ids"]
    )

    assert len(ds) == 4
    assert ds[0]["input_ids"] == [0, 1, 2, 3, 4, 5]
    assert ds[1]["input_ids"] == [6, 7, 8, 9, 10, 11, 12]
    assert ds[2]["input_ids"] == [13, 14, 15]
    assert ds[3]["input_ids"] == [16, 17, 18, 19]


@pytest.mark.parametrize(
    "block_size, expected_length, expected_output",
    [
        #(1, 20, [[i] for i in range(20)]), # TODO
        (2, 10, [[2 * i, 2 * i + 1] for i in range(10)]),
        (3, 6, [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(6)]),
        (10, 2, [list(range(10)), list(range(10, 20))]),
        (6, 3, [list(range(i * 6, i * 6 + 6)) for i in range(3)]),
        (20, 1, [list(range(20))]),
        (25, 0, []),
    ],
)
def test_packed_continuous_dataset_loading(
    dummy_packed_data_path, block_size, expected_length, expected_output
):
    ds = PackedMemMapDatasetContinuous(
        dummy_packed_data_path,
        sample_key="input_ids",
        block_size=block_size
    )
    assert len(ds) == expected_length
    retrieved_input_ids = [
        list(packed_samples["input_ids"])
        for packed_samples in ds
    ]
    assert retrieved_input_ids == expected_output


def test_packed_continuous_dataset_missing_file(dummy_packed_data_path):
    dummy_packed_data_path.unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        PackedMemMapDatasetContinuous(dummy_packed_data_path, block_size=10, sample_key="input_ids")


@pytest.mark.parametrize(
    "max_num_of_tokens, expected_index_size", [(None, 12), (10, 1)]
)
def test_create_packed_dataset(
    indexed_dummy_data_path,
    gpt2_tokenizer,
    max_num_of_tokens,
    expected_index_size
):
    block_size = 5
    packed_generator = PackedDataGenerator(
        src_path=indexed_dummy_data_path.raw_data_path,
        codecs={
            ".text": HfTokenizerCodec(
                tokenizer=gpt2_tokenizer,
            )
        },
        max_num_of_bytes=(
            (HfTokenizerCodec.TOKEN_SIZE_IN_BYTES * max_num_of_tokens)
            if max_num_of_tokens is not None else None
        )
    )
    default_packed_dataset_path = packed_generator._default_destination_path()
    assert not default_packed_dataset_path.is_file()
    packed_generator.run()
    packed_dataset = PackedMemMapDatasetContinuous(
        default_packed_dataset_path,
        sample_key="input_ids",
        block_size=block_size,
    )

    start_of_jsonl_content = "0 Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor"
    tokenized_start_of_jsonl_content = gpt2_tokenizer(start_of_jsonl_content)["input_ids"]
    packed_dataset_iterator = iter(packed_dataset)
    assert tokenized_start_of_jsonl_content[:block_size] == next(packed_dataset_iterator)["input_ids"]
    assert tokenized_start_of_jsonl_content[block_size : 2 * block_size] == next(packed_dataset_iterator)["input_ids"]
    assert len(packed_dataset._index_base) == expected_index_size

    # check validity of index section in packed dataset
    for idx, (offset, entry_length) in enumerate(packed_dataset._index_base[:-1]):
        assert offset + entry_length == packed_dataset._index_base[idx + 1][0]


def test_packed_image_dataset(indexed_dummy_image_data_path):
    # create packed data file
    packed_generator = PackedDataGenerator(
        src_path=indexed_dummy_image_data_path.raw_data_path,
        idx_path=indexed_dummy_image_data_path.index_path,
        codecs={
            ".img_path": PillowImageCodec()
        }
    )
    # get destination path
    default_packed_dataset_path = packed_generator._default_destination_path()
    assert not default_packed_dataset_path.is_file()
    # create packed dataset file
    packed_generator.run()

    # read dataset
    ds = PackedMemMapDataset(
        default_packed_dataset_path,
        sample_keys=["img"],
    )
    # read the jsonl to get the source image paths
    with indexed_dummy_image_data_path.raw_data_path.open("r") as f:
        src_data = list(map(json.loads, f.read().strip().split("\n")))
    # compare source image with dataset content
    for src, item in zip(src_data, ds):
        with Image.open(src["img_path"]) as src_img:
            numpy.testing.assert_allclose(src_img, item["img"])


def test_packed_audio_dataset(indexed_dummy_audio_data_path):
    # create packed data file
    packed_generator = PackedDataGenerator(
        src_path=indexed_dummy_audio_data_path.raw_data_path,
        idx_path=indexed_dummy_audio_data_path.index_path,
        codecs={".feat_path": TorchaudioAudioCodec()},
    )
    # get destination path
    default_packed_dataset_path = packed_generator._default_destination_path()
    assert not default_packed_dataset_path.is_file()
    # create packed dataset file
    packed_generator.run()

    # read dataset
    ds = PackedMemMapDataset(
        default_packed_dataset_path,
        sample_keys=["feat"],
    )
    # read the jsonl to get the source feature paths
    with indexed_dummy_audio_data_path.raw_data_path.open("r") as f:
        src_data = list(map(json.loads, f.read().strip().split("\n")))
    # compare source features with dataset content
    for src, item in zip(src_data, ds, strict=True):
        log_mel_spec = np.load(src["feat_path"])
        numpy.testing.assert_allclose(log_mel_spec, item["feat"])


def test_packed_multimodal_dataset(indexed_dummy_image_data_path, gpt2_tokenizer):
    # create packed data file
    packed_generator = PackedDataGenerator(
        src_path=indexed_dummy_image_data_path.raw_data_path,
        idx_path=indexed_dummy_image_data_path.index_path,
        codecs={
            ".img_path": PillowImageCodec(),
            ".text": HfTokenizerCodec(
                tokenizer=gpt2_tokenizer,
                add_eos_token=False
            )
        }
    )
    # get destination path
    default_packed_dataset_path = packed_generator._default_destination_path()
    assert not default_packed_dataset_path.is_file()
    # create packed dataset file
    packed_generator.run()

    # read dataset
    ds = PackedMemMapDataset(
        default_packed_dataset_path,
        sample_keys=["img", "input_ids"],
    )
    # read the jsonl to get the source values
    with indexed_dummy_image_data_path.raw_data_path.open("r") as f:
        src_data = list(map(json.loads, f.read().strip().split("\n")))
    # compare source with dataset content
    for src, item in zip(src_data, ds):
        with Image.open(src["img_path"]) as src_img:
            numpy.testing.assert_allclose(src_img, item["img"])
        assert gpt2_tokenizer(src["text"])["input_ids"] == item["input_ids"]
