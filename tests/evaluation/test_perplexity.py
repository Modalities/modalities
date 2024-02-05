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
    target_tensor = torch.tensor(
        [
            [0, 1, 2, 0, 1],
        ]
    )

    prediction_tensor = torch.tensor(
        [
            [
                [1.3151, -0.9029, 0.0504],
                [0.2887, 1.1838, -0.3253],
                [0.2163, 0.6919, -0.6849],
                [-0.6545, 0.1319, 0.8267],
                [-0.7220, -0.9223, -0.1635],
            ]
        ]
    )

    return InferenceResultBatch(
        targets={"target_ids": target_tensor}, predictions={"logits": prediction_tensor}, batch_dim=0
    )


@pytest.fixture
def expected_perplexity(batch_size_one_data: InferenceResultBatch) -> float:
    target_tensor = batch_size_one_data.targets["target_ids"]
    predictions = batch_size_one_data.predictions["logits"]

    all_log_probs = nn.LogSoftmax(dim=-1)(predictions)
    target_indices = target_tensor.unsqueeze(dim=-1)
    target_log_probs = torch.gather(all_log_probs, -1, target_indices)
    exponent = target_log_probs.sum() / target_log_probs.shape[-2] * -1.0
    return exponent.exp().item()


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_perplexity_computed_correctly_batch_size_one(
    aggregative_perplexity: AggregativePerplexity, batch_size_one_data: InferenceResultBatch, expected_perplexity: float
):
    aggregative_perplexity.add(batch_result=batch_size_one_data)
    perplexity = aggregative_perplexity.compute().item()
    assert perplexity == expected_perplexity


def test_perplexity_computed_correctly_batch_size_greater_one():
    ...
