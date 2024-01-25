import pytest
import torch

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import AsymmNCELoss


@pytest.fixture
def dummy_result_batch() -> InferenceResultBatch:
    predictions = {"embedding": torch.FloatTensor(1024, 512).uniform_(-50, 50)}
    targets = {"target": torch.zeros(1024, 512)}
    batch_dim = 1024
    result_batch = InferenceResultBatch(targets, predictions, batch_dim)
    return result_batch


# calculating asymmetric NCELoss between a batch of embeddings and itself --> zero
@pytest.mark.parametrize("key", ["embedding"])
def test_AsymmNCELoss_is_zero(dummy_result_batch, key):
    loss_func = AsymmNCELoss(key, key)
    assert loss_func(dummy_result_batch) <= 10e-6
