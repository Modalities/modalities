import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from modalities.dataloader.dataset import PackedMemMapDatasetBase
from modalities.dataloader.preprocessing.tokenization.tokenized_file_writer import TokenizedFileWriter


@pytest.mark.parametrize(
    "pbin_file_path, vocab_size",
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
