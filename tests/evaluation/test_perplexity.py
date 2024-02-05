import pytest
from torch import tensor

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
    target_tensor = tensor(
        [
            [0, 1, 2, 0, 1],
        ]
    )

    prediction_tensor = tensor(
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


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_perplexity_computed_correctly_batch_size_one(
    aggregative_perplexity: AggregativePerplexity, batch_size_one_data: InferenceResultBatch
):
    aggregative_perplexity.add(batch_result=batch_size_one_data)
    aggregative_perplexity.compute()
    ...
    # assert 1 == perplexity


def test_perplexity_computed_correctly_batch_size_greater_one():
    ...
