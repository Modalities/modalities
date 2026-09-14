import pytest
import torch

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import CLMCrossEntropyLoss, NCELoss, nce_loss


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


# ---------------------------------------------------------------------------
# Causal-LM cross-entropy must be computed in float32 even when the model hands
# it bfloat16 logits. FSDP2's MixedPrecisionPolicy casts parameters but does not
# install torch.autocast, so nothing promotes them on the way into the loss.
# ---------------------------------------------------------------------------


@pytest.fixture
def clm_loss() -> CLMCrossEntropyLoss:
    return CLMCrossEntropyLoss(target_key="target", prediction_key="logits")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_clm_cross_entropy_returns_float32_for_half_precision_logits(clm_loss, dtype):
    """The returned dtype is the giveaway: cross-entropy returns its input's dtype."""
    torch.manual_seed(0)
    logits = torch.randn(2, 16, 512, dtype=dtype)
    labels = torch.randint(0, 512, (2, 16))
    assert clm_loss(logits, labels).dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_clm_cross_entropy_is_accurate_for_half_precision_logits(clm_loss, dtype):
    """Half-precision logits must not drag the log-softmax or the reduction down with them.

    Two faults are guarded at once. Without the up-cast the log-softmax runs in the logits' own
    dtype, and because the reduction is ``"mean"`` the per-token losses are accumulated in that
    dtype too -- over a 131k-entry vocabulary the accumulator error dominates. Against a float64
    evaluation of the identical quantized logits, the un-upcast path lands around 5e-2 and the
    float32 path around 2e-6.
    """
    torch.manual_seed(0)
    vocab_size = 131072
    logits = (torch.randn(2, 64, vocab_size) * 8.0).to(dtype)
    labels = torch.randint(0, vocab_size, (2, 64))

    reference = clm_loss(logits.double(), labels)
    assert abs(clm_loss(logits, labels).double() - reference) < 1e-4


def test_clm_cross_entropy_matches_torchtitan_formulation(clm_loss):
    """Mean reduction over valid tokens == sum reduction / valid-token count.

    Mirrors ``torchtitan/components/loss.py::cross_entropy_loss``, which up-casts the logits and
    sum-reduces for token-based normalization. Kept inline so the test carries no dependency on
    torchtitan.
    """
    torch.manual_seed(0)
    vocab_size = 1024
    logits = torch.randn(2, 64, vocab_size, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (2, 64))
    labels[0, :7] = -100  # ignored tokens must leave both sides unchanged

    flat_labels = labels.view(-1).long()
    summed = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size).float(), flat_labels, reduction="sum", ignore_index=-100
    )
    torchtitan_style = summed / (flat_labels != -100).sum()

    torch.testing.assert_close(clm_loss(logits, labels), torchtitan_style, rtol=1e-6, atol=1e-6)
