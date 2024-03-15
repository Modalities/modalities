import pytest
import torch
import torch.nn as nn

from modalities.batch import InferenceResultBatch
from modalities.evaluation.perplexity import AggregativePerplexity
from tests.conftest import set_env_cpu


@pytest.fixture
def aggregative_perplexity() -> AggregativePerplexity:
    return AggregativePerplexity(
        prediction_key="logits",
        target_key="target_ids",
        local_rank=0,
    )


@pytest.fixture
def batch_size_one_data() -> InferenceResultBatch:
    # shape (batch_size, sequence_length)
    target_tensor = torch.tensor(
        [
            [2, 3],
        ]
    )

    # shape (batch_size, sequence_length, vocab_size)
    prediction_tensor = torch.tensor(
        [[[-0.7891, 1.3421, 0.4929, 0.0715, -0.0910], [0.9024, -0.8675, 0.8498, -1.0331, 0.5531]]]
    )

    return InferenceResultBatch(
        targets={"target_ids": target_tensor}, predictions={"logits": prediction_tensor}, batch_dim=0
    )


@pytest.fixture
def batch_size_two_data() -> InferenceResultBatch:
    # shape (batch_size, sequence_length)
    target_tensor = torch.tensor(
        [
            [2, 3, 2, 3],
            [1, 4, 0, 2],
        ]
    )

    # shape (batch_size, sequence_length, vocab_size)
    prediction_tensor = torch.tensor(
        [
            [
                [-0.7891, 1.3421, 0.4929, 0.0715, -0.0910],
                [0.9024, -0.8675, 0.8498, -1.0331, 0.5531],
                [-0.7891, 1.3421, 0.4929, 0.0715, -0.0910],
                [0.9024, -0.8675, 0.8498, -1.0331, 0.5531],
            ],
            [
                [1.3421, -0.7891, -0.0910, 0.8498, 0.0715],
                [0.9024, 0.9024, -1.0331, 0.5531, -0.0910],
                [-0.8675, -0.7891, 1.3421, 0.4929, 0.4929],
                [0.0715, 0.8498, -0.8675, -1.0331, 0.5531],
            ],
        ]
    )

    return InferenceResultBatch(
        targets={"target_ids": target_tensor}, predictions={"logits": prediction_tensor}, batch_dim=0
    )


def compute_average_expected_perplexity(batch: InferenceResultBatch) -> float:
    """Averages the perplexity for each sequence in the batch.
    Formula for perplexity computation of sequence of length $n$:
    $$
    perplexity(x_1, ..., x_n) = \exp \{ - \frac{1}{n} \sum_i^n \log p \left(x_i \middle| x_{<i} \right)\}
    $$

    :param batch: input batch with predictions and targets
    :type batch: InferenceResultBatch
    :return: sum of perplexities divided by batch size
    :rtype: float
    """
    sum_of_perplexities = 0.0
    for target_tensor, predictions in zip(batch.targets["target_ids"], batch.predictions["logits"]):
        all_log_probs = nn.LogSoftmax(dim=-1)(predictions)
        target_indices = target_tensor.unsqueeze(dim=-1)
        target_log_probs = torch.gather(all_log_probs, -1, target_indices)
        exponent = target_log_probs.sum() / target_log_probs.shape[-2]
        exponent = exponent * -1.0
        sum_of_perplexities += exponent.exp().item()
    return sum_of_perplexities / len(batch.targets["target_ids"])


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_perplexity_computed_correctly_batch_size_one(
    aggregative_perplexity: AggregativePerplexity, batch_size_one_data: InferenceResultBatch
):
    aggregative_perplexity.add(result_batch=batch_size_one_data)
    perplexity = aggregative_perplexity.compute().item()
    assert perplexity == pytest.approx(9.965, 0.01)
    assert perplexity == pytest.approx(compute_average_expected_perplexity(batch_size_one_data), 0.01)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_perplexity_computed_correctly_batch_size_greater_one(
    aggregative_perplexity: AggregativePerplexity, batch_size_two_data: InferenceResultBatch
):
    aggregative_perplexity.add(result_batch=batch_size_two_data)
    perplexity = aggregative_perplexity.compute().item()
    assert perplexity == pytest.approx(compute_average_expected_perplexity(batch_size_two_data), 0.01)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_sanity_check_computation():
    # Manual Sanity Check see: https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72
    # and https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # so this does not actually test the application, but rather is a semi-manual computation of the perplexity
    logits = torch.tensor([[-0.7891, 1.3421, 0.4929, 0.0715, -0.0910], [0.9024, -0.8675, 0.8498, -1.0331, 0.5531]])
    actual = torch.tensor([2, 3])
    as_batch = InferenceResultBatch(
        targets={"target_ids": actual.unsqueeze(0)}, predictions={"logits": logits.unsqueeze(0)}
    )
    loss_fun = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fun(logits, actual)
    loss = loss.sum() / len(actual)
    perplexity = torch.exp(loss)
    assert perplexity == pytest.approx(compute_average_expected_perplexity(as_batch), 0.01)
