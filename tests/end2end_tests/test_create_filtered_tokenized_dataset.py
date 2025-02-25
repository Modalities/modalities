import tempfile
from pathlib import Path
from typing import Callable

import pytest

from modalities.api import FileExistencePolicy, create_filtered_tokenized_dataset
from modalities.dataloader.dataset import PackedMemMapDatasetBase


@pytest.mark.parametrize(
    "input_data_path, file_existence_policy, filter_routine, expected_exception",
    [
        (
            Path("/raid/s3/opengptx/max_lue/repositories/modalities/tests/end2end_tests/lorem_ipsum.pbin"),
            FileExistencePolicy.ERROR,
            lambda x: True,  # take every sample
            False,
        ),
        (
            Path("/raid/s3/opengptx/max_lue/repositories/modalities/tests/end2end_tests/lorem_ipsum.pbin"),
            FileExistencePolicy.ERROR,
            lambda x: False,  # take no sample
            True,
        ),
        (
            Path("/raid/s3/opengptx/max_lue/repositories/modalities/tests/end2end_tests/lorem_ipsum.pbin"),
            FileExistencePolicy.ERROR,
            lambda x: x % 2 == 0,  # take every second sample
            False,
        ),
        (
            Path("/raid/s3/opengptx/max_lue/repositories/modalities/tests/end2end_tests/lorem_ipsum.pbin"),
            FileExistencePolicy.ERROR,
            lambda x: x == 2,  # take only the third sample
            False,
        ),
    ],
)
def test_create_filtered_tokenized_dataset(
    input_data_path: Path,
    file_existence_policy: FileExistencePolicy,
    filter_routine: Callable[[int], bool],
    expected_exception: bool,
):
    # we try out different filter routines to see if the filtering works as expected
    # by checking the original dataset and compare it to the filtered one.

    with tempfile.TemporaryDirectory() as temp_dir:
        output_data_path = Path(temp_dir) / "filtered_data.pbin"
        if expected_exception:
            with pytest.raises(ValueError):
                create_filtered_tokenized_dataset(
                    input_data_path=input_data_path,
                    filter_routine=filter_routine,
                    output_data_path=output_data_path,
                    file_existence_policy=file_existence_policy,
                )
            return
        else:
            create_filtered_tokenized_dataset(
                input_data_path=input_data_path,
                filter_routine=filter_routine,
                output_data_path=output_data_path,
                file_existence_policy=file_existence_policy,
            )

        sample_key = "text"
        dataset_original = PackedMemMapDatasetBase(
            raw_data_path=input_data_path, sample_key=sample_key, load_index=True
        )
        dataset_filtered = PackedMemMapDatasetBase(
            raw_data_path=output_data_path, sample_key=sample_key, load_index=True
        )

        assert len(dataset_original) > 0
        num_kept = 0
        for i in range(len(dataset_original)):
            if filter_routine(i):
                assert all(dataset_original[i][sample_key] == dataset_filtered[num_kept][sample_key])
                num_kept += 1

        assert num_kept == len(dataset_filtered)
