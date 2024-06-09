import pytest

from modalities.utils.number_conversion import NumberConversion


@pytest.mark.parametrize(
    "num_ranks,global_num_samples,expected",
    [(2, 10, 5), (2, 11, 5)],
)
def test_get_local_num_batches_from_num_samples(num_ranks: int, global_num_samples: int, expected: int):
    assert NumberConversion.get_local_num_batches_from_num_samples(num_ranks, global_num_samples) == expected


@pytest.mark.parametrize(
    "num_ranks,global_num_tokens,context_size,expected",
    [(2, 10, 2, 2), (2, 11, 2, 2), (2, 12, 2, 3)],
)
def test_get_local_num_batches_from_num_tokens(
    num_ranks: int, global_num_tokens: int, context_size: int, expected: int
):
    assert (
        NumberConversion.get_local_num_batches_from_num_tokens(num_ranks, global_num_tokens, context_size) == expected
    )


@pytest.mark.parametrize(
    "num_ranks,local_micro_batch_size,global_num_samples,expected",
    [(2, 2, 10, 2), (2, 2, 11, 2), (2, 2, 12, 3)],
)
def test_get_num_steps_from_num_samples(
    num_ranks: int, local_micro_batch_size: int, global_num_samples: int, expected: int
):
    assert (
        NumberConversion.get_num_steps_from_num_samples(num_ranks, local_micro_batch_size, global_num_samples)
        == expected
    )


@pytest.mark.parametrize(
    "num_ranks,local_micro_batch_size,global_num_tokens,context_size,expected",
    [(2, 2, 20, 2, 2), (2, 2, 21, 2, 2), (2, 2, 22, 2, 2), (2, 2, 24, 2, 3)],
)
def test_get_num_steps_from_num_tokens(
    num_ranks: int, local_micro_batch_size: int, global_num_tokens: int, context_size: int, expected: int
):
    assert (
        NumberConversion.get_num_steps_from_num_tokens(
            num_ranks, local_micro_batch_size, global_num_tokens, context_size
        )
        == expected
    )


@pytest.mark.parametrize(
    "num_ranks,local_micro_batch_size,context_size,num_steps_done,expected",
    [
        (2, 2, 2, 2, 16),
        (2, 2, 2, 3, 24),
    ],
)
def test_get_num_tokens_from_num_steps_callable(
    num_ranks: int, local_micro_batch_size: int, context_size: int, num_steps_done: int, expected: int
):
    assert (
        NumberConversion.get_num_tokens_from_num_steps_callable(num_ranks, local_micro_batch_size, context_size)(
            num_steps_done
        )
        == expected
    )
