import pytest
import torch

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import NCELoss, nce_loss


@pytest.fixture
def dummy_result_batch() -> InferenceResultBatch:
    predictions = {"embedding": torch.FloatTensor(1024, 512).uniform_(-50, 50)}
    targets = {"target": torch.zeros(1024, 512)}
    batch_dim = 1024
    result_batch = InferenceResultBatch(targets, predictions, batch_dim)
    return result_batch


# calculating asymmetric NCELoss between a batch of embeddings and itself --> zero
@pytest.mark.parametrize("key", ["embedding"])
def test_asymm_NCELoss_is_zero(dummy_result_batch, key):
    loss_func = NCELoss(prediction_key1=key, prediction_key2=key)
    assert loss_func(dummy_result_batch) <= 10e-6


# calculating nce_loss for two randomly generated batch of embeddings --> 1.1300 (pre-calculated manually)
@pytest.mark.parametrize(
    "embedding1,embedding2",
    [
        (
            torch.Tensor([[0.38, 0.18], [0.36, 0.66], [0.72, 0.09]]),
            torch.Tensor([[0.48, 0.01], [0.54, 0.28], [0.08, 0.34]]),
        )
    ],
)
def test_nce_loss_correctness(embedding1, embedding2):
    loss = nce_loss(embedding1, embedding2, device="cpu", is_asymmetric=True, temperature=1.0)
    assert loss == pytest.approx(1.1300, 0.0001)
