import pytest
import torch

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import CLMCrossEntropyLoss, LossFactory, NCELoss, nce_loss


@pytest.fixture
def dummy_result_batch() -> InferenceResultBatch:
    predictions = {"embedding": torch.rand(1024, 512)}
    targets = {"target": torch.zeros(1024, 512)}
    batch_dim = 1024
    result_batch = InferenceResultBatch(targets, predictions, batch_dim)
    return result_batch


# calculating asymmetric NCELoss between a batch of embeddings and itself --> zero
@pytest.mark.parametrize("key", ["embedding"])
def test_asymm_NCELoss_is_zero(dummy_result_batch, key):
    loss_func = NCELoss(prediction_key1=key, prediction_key2=key)
    assert loss_func(dummy_result_batch) <= 10e-6


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


def test_clm_cross_entropy_loss_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(4, 8, 16)  # (batch, sequence, vocab)
    targets = torch.randint(0, 16, (4, 8))

    loss_func = CLMCrossEntropyLoss(target_key="target", prediction_key="logits")
    actual = loss_func(logits, targets)

    expected = torch.nn.functional.cross_entropy(logits.view(-1, 16), targets.view(-1), reduction="mean")
    assert torch.allclose(actual, expected)


def test_clm_cross_entropy_loss_ignores_masked_tokens():
    # tokens labeled with -100 must be excluded from the loss (used for loss masking)
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    masked_targets = targets.clone()
    masked_targets[:, 2:] = -100

    loss_func = CLMCrossEntropyLoss(target_key="target", prediction_key="logits")
    actual = loss_func(logits, masked_targets)

    # equals the loss computed only over the unmasked tokens
    expected = torch.nn.functional.cross_entropy(
        logits[:, :2].reshape(-1, 8), targets[:, :2].reshape(-1), reduction="mean"
    )
    assert torch.allclose(actual, expected)


def test_get_compiled_loss_matches_eager():
    torch.manual_seed(0)
    logits = torch.randn(4, 8, 16)
    targets = torch.randint(0, 16, (4, 8))

    expected = CLMCrossEntropyLoss(target_key="target", prediction_key="logits")(logits, targets)

    loss = CLMCrossEntropyLoss(target_key="target", prediction_key="logits")
    compiled_loss = LossFactory.get_compiled_loss(loss, fullgraph=True)

    # the loss is compiled in place and returned, preserving its identity and interface
    assert compiled_loss is loss
    assert compiled_loss.tag == "CLMCrossEntropyLoss"
    assert compiled_loss.target_key == "target"

    actual = compiled_loss(logits, targets)
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_get_compiled_loss_raises_for_unsupported_loss():
    loss = NCELoss(prediction_key1="embedding", prediction_key2="embedding")
    with pytest.raises(NotImplementedError):
        LossFactory.get_compiled_loss(loss, fullgraph=True)
