import pickle
from pathlib import Path

import pytest

from modalities.dataloader.dataset_factory import DatasetFactory
from modalities.utils.number_conversion import NumberConversion


@pytest.mark.parametrize(
    "num_ranks,global_num_samples,local_micro_batch_size,expected",
    [(2, 100, 10, 5), (2, 110, 10, 5), (4, 100, 10, 2), (4, 100, 5, 5)],
)
def test_get_local_num_batches_from_num_samples(
    num_ranks: int, global_num_samples: int, local_micro_batch_size: int, expected: int
):
    assert (
        NumberConversion.get_local_num_batches_from_num_samples(num_ranks, global_num_samples, local_micro_batch_size)
        == expected
    )


@pytest.mark.parametrize(
    "num_ranks,global_num_tokens,sequence_length,local_micro_batch_size,expected",
    [(2, 100, 2, 10, 2), (2, 110, 2, 10, 2), (2, 120, 2, 10, 3), (4, 100, 3, 4, 2)],
)
def test_get_local_num_batches_from_num_tokens(
    num_ranks: int, global_num_tokens: int, sequence_length: int, local_micro_batch_size: int, expected: int
):
    assert (
        NumberConversion.get_local_num_batches_from_num_tokens(
            num_ranks, global_num_tokens, sequence_length, local_micro_batch_size
        )
        == expected
    )


@pytest.mark.parametrize(
    "num_ranks,local_micro_batch_size,global_num_samples,gradient_accumulation_steps,expected",
    [(2, 2, 10, 1, 2), (2, 2, 11, 1, 2), (2, 2, 12, 1, 3), (2, 2, 20, 2, 2), (2, 2, 22, 2, 2), (2, 2, 48, 4, 3)],
)
def test_get_num_steps_from_num_samples(
    num_ranks: int,
    local_micro_batch_size: int,
    global_num_samples: int,
    gradient_accumulation_steps: int,
    expected: int,
):
    assert (
        NumberConversion.get_num_steps_from_num_samples(
            num_ranks, local_micro_batch_size, global_num_samples, gradient_accumulation_steps
        )
        == expected
    )


@pytest.mark.parametrize(
    "num_ranks,local_micro_batch_size,global_num_tokens,sequence_length,gradient_accumulation_steps,expected",
    [
        (2, 2, 20, 2, 1, 2),
        (2, 2, 21, 2, 1, 2),
        (2, 2, 22, 2, 1, 2),
        (2, 2, 24, 2, 1, 3),
        (2, 2, 40, 2, 2, 2),
        (2, 2, 42, 2, 2, 2),
        (2, 2, 88, 2, 4, 2),
        (2, 2, 48, 2, 2, 3),
    ],
)
def test_get_num_steps_from_num_tokens(
    num_ranks: int,
    local_micro_batch_size: int,
    global_num_tokens: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
    expected: int,
):
    assert (
        NumberConversion.get_num_steps_from_num_tokens(
            num_ranks, local_micro_batch_size, global_num_tokens, sequence_length, gradient_accumulation_steps
        )
        == expected
    )


@pytest.mark.parametrize(
    "num_steps,num_ranks,local_micro_batch_size,sequence_length,gradient_accumulation_steps,expected",
    [
        (2, 3, 20, 2, 1, 240),
        (2, 3, 21, 2, 1, 252),
        (3, 4, 88, 2, 4, 8448),
        (3, 4, 48, 2, 2, 2304),
    ],
)
def test_get_num_tokens_from_num_steps(
    num_steps: int,
    num_ranks: int,
    local_micro_batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
    expected: int,
):
    assert (
        NumberConversion.get_num_tokens_from_num_steps(
            num_steps=num_steps,
            num_ranks=num_ranks,
            local_micro_batch_size=local_micro_batch_size,
            sequence_length=sequence_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        == expected
    )


@pytest.mark.parametrize(
    "checkpoint_path,expected,expected_exception",
    [
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-model-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            249,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            249,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/seen_steps_1234-eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            None,
            ValueError,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-abc_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            None,
            ValueError,
        ),
    ],
)
def test_get_last_step_from_checkpoint_path(checkpoint_path: Path, expected: int, expected_exception: Exception):
    if expected_exception:
        # Expecting an exception for this test case
        with pytest.raises(expected_exception):
            NumberConversion.get_last_step_from_checkpoint_path(checkpoint_path=checkpoint_path)
    else:
        assert NumberConversion.get_last_step_from_checkpoint_path(checkpoint_path=checkpoint_path) == expected


@pytest.mark.parametrize(
    "checkpoint_path,expected,expected_exception",
    [
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-model-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            250,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            250,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/seen_steps_1234-eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            None,
            ValueError,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-abc_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            None,
            ValueError,
        ),
    ],
)
def test_get_num_seen_steps_from_checkpoint_path(checkpoint_path: Path, expected: int, expected_exception: Exception):
    if expected_exception:
        # Expecting an exception for this test case
        with pytest.raises(expected_exception):
            NumberConversion.get_num_seen_steps_from_checkpoint_path(checkpoint_path=checkpoint_path)
    else:
        assert NumberConversion.get_num_seen_steps_from_checkpoint_path(checkpoint_path=checkpoint_path) == expected


@pytest.mark.parametrize(
    "checkpoint_path,expected,expected_exception",
    [
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-model-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            65536000,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            65536000,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/seen_tokens_65-eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            None,
            ValueError,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-abc_65536000-target_tokens_1310720000.bin",
            None,
            ValueError,
        ),
    ],
)
def test_get_global_num_seen_tokens_from_checkpoint_path(
    checkpoint_path: Path, expected: int, expected_exception: Exception
):
    if expected_exception:
        # Expecting an exception for this test case
        with pytest.raises(expected_exception):
            NumberConversion.get_global_num_seen_tokens_from_checkpoint_path(checkpoint_path=checkpoint_path)
    else:
        assert (
            NumberConversion.get_global_num_seen_tokens_from_checkpoint_path(checkpoint_path=checkpoint_path)
            == expected
        )


@pytest.mark.parametrize(
    "checkpoint_path,expected,expected_exception",
    [
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-model-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            1310720000,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            1310720000,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/target_tokens_65-eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            None,
            ValueError,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-abc_65536000-abc_1310720000.bin",
            None,
            ValueError,
        ),
    ],
)
def test_get_global_num_target_tokens_from_checkpoint_path(
    checkpoint_path: Path, expected: int, expected_exception: Exception
):
    if expected_exception:
        # Expecting an exception for this test case
        with pytest.raises(expected_exception):
            NumberConversion.get_global_num_target_tokens_from_checkpoint_path(checkpoint_path=checkpoint_path)
    else:
        assert (
            NumberConversion.get_global_num_target_tokens_from_checkpoint_path(checkpoint_path=checkpoint_path)
            == expected
        )


@pytest.mark.parametrize(
    "checkpoint_path,expected,expected_exception",
    [
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-model-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            5000,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            5000,
            None,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/target_tokens_65-eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-seen_tokens_65536000-target_tokens_1310720000.bin",
            None,
            ValueError,
        ),
        (
            "/checkpoints/2024-09-09__21-19-55_b7580cd4/eid_2024-09-09__21-19-55_b7580cd4-optimizer-seen_steps_250-abc_65536000-abc_1310720000.bin",
            None,
            ValueError,
        ),
    ],
)
def test_get_num_target_steps_from_checkpoint_path(checkpoint_path: Path, expected: int, expected_exception: Exception):
    if expected_exception:
        # Expecting an exception for this test case
        with pytest.raises(expected_exception):
            NumberConversion.get_num_target_steps_from_checkpoint_path(checkpoint_path=checkpoint_path)
    else:
        assert NumberConversion.get_num_target_steps_from_checkpoint_path(checkpoint_path=checkpoint_path) == expected


@pytest.mark.parametrize(
    "dataset_path,sequence_length,num_ranks,local_micro_batch_size,gradient_accumulation_steps",
    [
        (
            Path("tests/end2end_tests/lorem_ipsum.pbin"),
            256,
            2,
            2,
            2,
        ),
        (
            Path("tests/end2end_tests/lorem_ipsum.pbin"),
            32,
            2,
            2,
            2,
        ),
        (
            Path("tests/end2end_tests/lorem_ipsum.pbin"),
            128,
            2,
            2,
            2,
        ),
    ],
)
def test_get_num_tokens_from_packed_mem_map_dataset_continuous(
    dataset_path: Path,
    sequence_length: int,
    num_ranks: int,
    local_micro_batch_size: int,
    gradient_accumulation_steps: int,
):
    dataset = DatasetFactory.get_packed_mem_map_dataset_continuous(
        raw_data_path=dataset_path, sequence_length=sequence_length, sample_key="text"
    )

    max_num_tokens = len(dataset._index) * sequence_length
    num_steps = max_num_tokens // sequence_length // num_ranks // local_micro_batch_size // gradient_accumulation_steps
    effective_num_tokens = (
        num_steps * num_ranks * local_micro_batch_size * gradient_accumulation_steps * sequence_length
    )

    assert (
        NumberConversion.get_num_tokens_from_packed_mem_map_dataset_continuous(
            dataset_path=dataset_path,
            sequence_length=sequence_length,
            num_ranks=num_ranks,
            local_micro_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        == effective_num_tokens
    )


@pytest.mark.parametrize(
    "num_ranks,local_micro_batch_size,gradient_accumulation_steps",
    [
        (2, 3, 2),
        (2, 3, 2),
        (3, 4, 2),
        (3, 4, 2),
    ],
)
def test_num_steps_from_raw_dataset_index(
    num_ranks: int, local_micro_batch_size: int, gradient_accumulation_steps: int
):
    working_dir = Path(__file__).parent
    raw_dataset_path = working_dir / "../../data/lorem_ipsum_long.jsonl"
    raw_index_path = working_dir / "../../data/lorem_ipsum_long.idx"

    with open(raw_dataset_path, "r") as f:
        num_samples = len(f.readlines())

    with open(raw_index_path, "rb") as f:
        index_length = len(pickle.load(f))

    num_steps_from_number_conversion = NumberConversion.get_num_steps_from_raw_dataset_index(
        raw_index_path=raw_index_path,
        num_ranks=num_ranks,
        local_micro_batch_size=local_micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    assert num_samples == index_length
    assert (
        num_steps_from_number_conversion
        == index_length // num_ranks // local_micro_batch_size // gradient_accumulation_steps
    )
