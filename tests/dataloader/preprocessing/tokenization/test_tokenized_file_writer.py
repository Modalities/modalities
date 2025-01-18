import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from modalities.dataloader.dataset import PackedMemMapDatasetBase
from modalities.dataloader.preprocessing.tokenization.tokenized_file_writer import TokenizedFileWriter


@pytest.mark.parametrize(
    "pbin_file_path, vocab_size, num_documents",
    [
        (Path("tests/data/datasets/lorem_ipsum_long.pbin"), 50257, 500),
    ],
)
def test_write_tokenized_dataset_via_existing_pbin_file(pbin_file_path: Path, vocab_size: int, num_documents: int):
    sample_key = "text"
    dataset = PackedMemMapDatasetBase(raw_data_path=pbin_file_path, sample_key=sample_key, load_index=True)

    in_memory_dataset: list[np.ndarray] = dataset[:][sample_key]
    assert len(in_memory_dataset) == num_documents
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file_path = Path(temp_file.name)
        TokenizedFileWriter.write_tokenized_dataset(
            tokenized_dataset=in_memory_dataset, tokenized_dataset_file_path=temp_file_path, vocab_size=vocab_size
        )

        # hash both files
        with open(pbin_file_path, "rb") as f:
            orig_pbin_file_hash = hashlib.md5(f.read()).hexdigest()
        with open(temp_file_path, "rb") as f:
            new_pbin_file_hash = hashlib.md5(f.read()).hexdigest()

    assert orig_pbin_file_hash == new_pbin_file_hash


@pytest.mark.parametrize(
    "dataset, vocab_size, expect_error",
    [
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])], 10, False),
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 255])], 16, False),
        # 256 cannote be represented by 1 byte
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 256])], 16, True),
        # gpt2 tokenizer has 50257 tokens
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 50256])], 50257, False),
        # even though the last token is 65535, it can be represented by 2 bytes
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 65535])], 50257, False),
        # 65536 cannot be represented by 2 bytes
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 65536])], 50257, True),
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 65537])], 65536, True),
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 65536])], 65536, True),
        # if we increase the vocab size, the last token can be represented by 4 bytes
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 65536])], 65537, False),
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 65537])], 65537, False),
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 65538])], 65537, False),
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 4294967295])], 65537, False),
        # 4294967296 cannot be represented by 4 bytes
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 4294967296])], 65537, True),
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 4294967296])], 4294967296, True),
        # we only support tokens of up to 4 bytes i.e., 0 to 4294967295
        ([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 4294967296])], 4294967297, True),
    ],
)
def test_write_tokenized_dataset(dataset: list[np.ndarray], vocab_size: int, expect_error: bool):
    sample_key = "text"
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file_path = Path(temp_file.name)
        if expect_error:
            with pytest.raises(ValueError):
                TokenizedFileWriter.write_tokenized_dataset(
                    tokenized_dataset=dataset, tokenized_dataset_file_path=temp_file_path, vocab_size=vocab_size
                )
            return
        else:
            TokenizedFileWriter.write_tokenized_dataset(
                tokenized_dataset=dataset, tokenized_dataset_file_path=temp_file_path, vocab_size=vocab_size
            )

        new_dataset = PackedMemMapDatasetBase(raw_data_path=temp_file_path, sample_key=sample_key, load_index=True)[:][
            sample_key
        ]

    assert all([all(d1 == d2) for d1, d2 in zip(dataset, new_dataset)])
