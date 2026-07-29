import pytest
import torch
import torch.nn.functional as F

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import ChunkedCLMCrossEntropyLoss, NCELoss, nce_loss

BATCH_SIZE = 2
SEQ_LEN = 16
VOCAB_SIZE = 64
PREDICTION_KEY = "logits"
TARGET_KEY = "labels"


@pytest.fixture
def dummy_result_batch() -> InferenceResultBatch:
    predictions = {"embedding": torch.rand(1024, 512)}
    targets = {"target": torch.zeros(1024, 512)}
    batch_dim = 1024
    result_batch = InferenceResultBatch(targets, predictions, batch_dim)
    return result_batch


@pytest.fixture
def clm_batch() -> InferenceResultBatch:
    torch.manual_seed(42)
    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    return InferenceResultBatch(
        targets={TARGET_KEY: labels},
        predictions={PREDICTION_KEY: logits},
        batch_dim=BATCH_SIZE,
    )


# calculating asymmetric NCELoss between a batch of embeddings and itself --> zero
@pytest.mark.parametrize("key", ["embedding"])
def test_asymm_NCELoss_is_zero(dummy_result_batch, key):
    loss_func = NCELoss(prediction_key1=key, prediction_key2=key)
    assert loss_func(dummy_result_batch) <= 10e-6


@pytest.mark.parametrize("num_chunks,use_compile", [(1, False), (1, True), (4, False), (4, True)])
def test_chunked_clm_cross_entropy_matches_reference(clm_batch, num_chunks, use_compile):
    """ChunkedCLMCrossEntropyLoss must match plain F.cross_entropy for all chunk/compile combos."""
    loss_fn = ChunkedCLMCrossEntropyLoss(
        target_key=TARGET_KEY,
        prediction_key=PREDICTION_KEY,
        num_chunks=num_chunks,
        use_compile=use_compile,
    )
    result = loss_fn(clm_batch).item()

    logits = clm_batch.get_predictions(PREDICTION_KEY)
    labels = clm_batch.get_targets(TARGET_KEY)
    reference = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1).long(), reduction="mean").item()

    assert result == pytest.approx(reference, rel=1e-4)


def test_chunked_clm_cross_entropy_invalid_num_chunks():
    with pytest.raises(ValueError, match="num_chunks must be >= 1"):
        ChunkedCLMCrossEntropyLoss(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY, num_chunks=0)


def test_chunked_clm_cross_entropy_tag():
    loss_fn = ChunkedCLMCrossEntropyLoss(target_key=TARGET_KEY, prediction_key=PREDICTION_KEY)
    assert loss_fn.tag == "ChunkedCLMCrossEntropyLoss"


# calculating nce_loss for two randomly generated batch of embeddings (manually calculated)
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
    unidirectional_loss = nce_loss(embedding1, embedding2, device="cpu", is_asymmetric=True, temperature=1.0)
    bidirectional_loss = nce_loss(embedding1, embedding2, device="cpu", is_asymmetric=False, temperature=1.0)
    assert unidirectional_loss == pytest.approx(1.1300, 0.0001)
    assert bidirectional_loss == pytest.approx(2.2577, 0.0001)
