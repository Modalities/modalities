import hashlib
import tempfile
from pathlib import Path

import pytest

from modalities.api import FileExistencePolicy, shuffle_tokenized_data
from modalities.dataloader.dataset import PackedMemMapDatasetBase


def _calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@pytest.mark.parametrize(
    "tokenized_data_file_path, batch_size",
    [
        (Path("tests/end2end_tests/lorem_ipsum.pbin"), 2),
    ],
)
def test_shuffle_tokenized_data(tokenized_data_file_path: Path, batch_size: int):
    # temporary file
    md5sums = []
    seeds = [1, 1, 2]
    file_paths = []
    datasets = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(3):
            temp_file = Path(temp_dir) / f"shuffled_data_{i}.pbin"
            file_paths.append(temp_file)
            shuffle_tokenized_data(
                tokenized_data_file_path,
                output_data_path=temp_file,
                batch_size=batch_size,
                file_existence_policy=FileExistencePolicy.OVERRIDE,
                seed=seeds[i],
            )
            md5sums.append(_calculate_md5(temp_file))
            datasets.append(PackedMemMapDatasetBase(raw_data_path=temp_file, sample_key="text", load_index=True))

        # check that the different seeds lead to different orderings
        # and that the same seed leads to the same ordering
        assert md5sums[0] == md5sums[1]
        assert md5sums[0] != md5sums[2]

        assert len(datasets[0]) == len(datasets[1]) == len(datasets[2])
        for i in range(len(datasets[0])):
            assert all(datasets[0][i]["text"] == datasets[1][i]["text"])

        # when we shuffle some lines might end up in the same place
        # in this test we make sure that at least one line is at a different place
        num_differing_lines = 0
        for i in range(len(datasets[0])):
            if len(datasets[0][i]["text"]) == len(datasets[2][i]["text"]):
                num_differing_lines += int(any(datasets[0][i]["text"] != datasets[2][i]["text"]))
            else:
                num_differing_lines += 1
        assert num_differing_lines > 0
