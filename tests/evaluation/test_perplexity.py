import pytest
import torch

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
            [2, 3],
        ]
    )

    prediction_tensor = torch.tensor(
        [[-0.7891, 1.3421, 0.4929, 0.0715, -0.0910], [0.9024, -0.8675, 0.8498, -1.0331, 0.5531]]
    )

    return InferenceResultBatch(
        targets={"target_ids": target_tensor}, predictions={"logits": prediction_tensor}, batch_dim=0
    )


@pytest.fixture
def batch_size_two_data() -> InferenceResultBatch:
    target_tensor = torch.tensor(
        [
            [2, 3, 2, 3],
        ]
    )

    prediction_tensor = torch.tensor(
        [
            [-0.7891, 1.3421, 0.4929, 0.0715, -0.0910],
            [0.9024, -0.8675, 0.8498, -1.0331, 0.5531],
            [-0.7891, 1.3421, 0.4929, 0.0715, -0.0910],
            [0.9024, -0.8675, 0.8498, -1.0331, 0.5531],
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
    perplexity = aggregative_perplexity.compute().item()
    assert 9.965 == pytest.approx(perplexity, 0.01)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_perplexity_computed_correctly_batch_size_greater_one(
    aggregative_perplexity: AggregativePerplexity, batch_size_two_data: InferenceResultBatch
):
    aggregative_perplexity.add(batch_result=batch_size_two_data)
    perplexity = aggregative_perplexity.compute().item()
    assert 9.965 == pytest.approx(perplexity, 0.01)
    pytest.fail("Isn't the test test_perplexity_computed_correctly_batch_size_one not alreay batch size 2?")


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_sanity_check_computation():
    # Manual Sanity Check see: https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72
    # and https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # so this does not actually test the application, but rather is a semi-manual computation of the perplexity
    logits = torch.tensor([[-0.7891, 1.3421, 0.4929, 0.0715, -0.0910], [0.9024, -0.8675, 0.8498, -1.0331, 0.5531]])
    actual = torch.tensor([2, 3])
    loss_fun = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fun(logits, actual)
    loss = loss.sum() / len(actual)
    perplexity = torch.exp(loss)
    assert 9.965 == pytest.approx(perplexity, 0.01)
