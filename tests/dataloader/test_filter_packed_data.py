import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from modalities.dataloader.dataset import PackedMemMapDatasetBase
from modalities.dataloader.filter_packed_data import filter_dataset


def test_creates_output_file(tmp_path: Path, packed_data_path: Path):
    output_path = Path(tmp_path, "output.pbin")
    filter_dataset(
        src_path=packed_data_path, dst_path=output_path, filter_func=accept_even_indices, sample_key="input_ids"
    )
    assert output_path.exists()


def test_filtered_data_has_expected_length(tmp_path: Path, packed_data_path: Path):
    output_path = Path(tmp_path, "output.pbin")
    filter_dataset(
        src_path=packed_data_path, dst_path=output_path, filter_func=accept_even_indices, sample_key="input_ids"
    )
    original_data = PackedMemMapDatasetBase(packed_data_path, sample_key="input_ids")
    filtered_data = PackedMemMapDatasetBase(output_path, sample_key="input_ids")
    assert (
        len(filtered_data) == len(original_data) // 2 + len(original_data) % 2
    ), "Filtered data length should be half of the original data length (rounded up)."


def test_filtered_data_has_expected_content(tmp_path: Path, dummy_packed_data_path: Path):
    output_path = Path(tmp_path, "output.pbin")
    filter_dataset(
        src_path=dummy_packed_data_path, dst_path=output_path, filter_func=accept_even_indices, sample_key="input_ids"
    )
    filtered_data = PackedMemMapDatasetBase(output_path, sample_key="input_ids")
    assert filtered_data[0]["input_ids"].tolist() == list(range(24 // 4))
    assert filtered_data[1]["input_ids"].tolist() == list(range(64 // 4, (64 + 12) // 4))


def test_always_true_filtered_data_has_identical_file_hash(tmp_path: Path, packed_data_path: Path):
    output_path = Path(tmp_path, "output.pbin")
    filter_dataset(src_path=packed_data_path, dst_path=output_path, filter_func=lambda x: True, sample_key="input_ids")
    with open(packed_data_path, "rb") as f_in, open(output_path, "rb") as f_out:
        original_hash = hashlib.sha256(f_in.read()).hexdigest()
        filtered_hash = hashlib.sha256(f_out.read()).hexdigest()
    assert (
        original_hash == filtered_hash
    ), "Filtered data should have the same hash as the original data when no filtering is applied."


def test_always_false_filtered_data_produces_valid_file(tmp_path: Path, packed_data_path: Path):
    output_path = Path(tmp_path, "output.pbin")
    filter_dataset(src_path=packed_data_path, dst_path=output_path, filter_func=lambda x: False, sample_key="input_ids")
    filtered_data = PackedMemMapDatasetBase(output_path, sample_key="input_ids")
    assert len(filtered_data) == 0, "Filtered data should be empty when all samples are filtered out."
    assert output_path.stat().st_size > 0, "Output file should not be empty even if no samples are included."


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


@pytest.fixture(params=[0, 1])
def packed_data_path(dummy_packed_data_path: Path, request: pytest.FixtureRequest) -> Path:
    path_options = [dummy_packed_data_path, Path("tests/data/datasets/lorem_ipsum_long.pbin")]
    return path_options[request.param]
