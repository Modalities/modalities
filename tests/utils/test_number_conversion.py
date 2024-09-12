import pytest

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
