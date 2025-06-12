from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from modalities.dataloader.dataset import PackedMemMapDatasetBase
from modalities.dataloader.filter_packed_data import filter_dataset


def test_creates_output_file(tmp_path: Path, dummy_packed_data_path: Path):
    output_path = Path(tmp_path, "output.pbin")
    filter_dataset(
        dst_path=output_path, src_path=dummy_packed_data_path, filter_func=accept_even_indices, sample_key="input_ids"
    )
    assert output_path.exists()


def test_filtered_data_has_expected_length(tmp_path: Path, dummy_packed_data_path: Path):
    output_path = Path(tmp_path, "output.pbin")
    filter_dataset(
        dst_path=output_path, src_path=dummy_packed_data_path, filter_func=accept_even_indices, sample_key="input_ids"
    )
    filtered_data = PackedMemMapDatasetBase(output_path, sample_key="input_ids")
    assert len(filtered_data) == 2


def test_filtered_data_has_expected_content(tmp_path: Path, dummy_packed_data_path: Path):
    output_path = Path(tmp_path, "output.pbin")
    filter_dataset(
        dst_path=output_path, src_path=dummy_packed_data_path, filter_func=accept_even_indices, sample_key="input_ids"
    )
    filtered_data = PackedMemMapDatasetBase(output_path, sample_key="input_ids")
    assert filtered_data[0]["input_ids"].tolist() == list(range(24 // 4))
    assert filtered_data[1]["input_ids"].tolist() == list(range(64 // 4, (64 + 12) // 4))


def accept_even_indices(idx_content: tuple[int, dict[str, NDArray[np.int_]]]) -> bool:
    """
    Filter function that accepts only even indices.
    Args:
        idx (int): The index of the sample.
        content (Any): The content of the sample (not used in this filter).
    Returns:
        bool: True if the index is even, False otherwise.
    """
    idx, _ = idx_content
    return idx % 2 == 0
