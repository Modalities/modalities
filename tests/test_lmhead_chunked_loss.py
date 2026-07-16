import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import IGNORE_INDEX, ChunkedLMHeadCrossEntropyLoss

BATCH_SIZE = 2
SEQ_LEN = 16
VOCAB_SIZE = 64
HIDDEN_DIM = 32


class TinyTrunk(nn.Module):
    """Stand-in for the transformer trunk: embedding + linear, produces hidden states."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.wte(inputs))


class TinyModelWithLMHead(nn.Module):
    """Mimics the GPT2LLM module structure (transformer.lm_head) for bind_lm_head."""

    def __init__(self, lm_head: nn.Module):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(lm_head=lm_head))


def _make_modules_and_data(
    seed: int = 42, mask_some_labels: bool = False
) -> tuple[TinyTrunk, nn.Linear, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    trunk = TinyTrunk()
    lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
    inputs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    if mask_some_labels:
        labels[0, : SEQ_LEN // 2] = IGNORE_INDEX
    return trunk, lm_head, inputs, labels


def _reference_loss_and_grads(
    trunk: TinyTrunk, lm_head: nn.Linear, inputs: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unchunked reference: full logits, CE with mean reduction, single backward."""
    hidden = trunk(inputs)
    logits = lm_head(hidden)
    loss = F.cross_entropy(
        logits.view(-1, VOCAB_SIZE), labels.view(-1), reduction="mean", ignore_index=IGNORE_INDEX
    )
    loss.backward()
    grads = {name: p.grad.clone() for name, p in trunk.named_parameters()}
    grads["lm_head.weight"] = lm_head.weight.grad.clone()
    return loss.detach(), grads


def _chunked_loss_and_grads(
    trunk: TinyTrunk,
    lm_head: nn.Linear,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_chunks: int,
    grad_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_fn = ChunkedLMHeadCrossEntropyLoss(
        target_key="target", prediction_key="prediction", num_chunks=num_chunks, use_compile=False
    )
    loss_fn.lm_head = lm_head
    hidden = trunk(inputs)
    loss = loss_fn.compute_and_backward(hidden_states=hidden, labels=labels, grad_scale=grad_scale)
    grads = {name: p.grad.clone() for name, p in trunk.named_parameters()}
    grads["lm_head.weight"] = lm_head.weight.grad.clone()
    return loss, grads


@pytest.mark.parametrize("num_chunks", [1, 2, 4, 8])
@pytest.mark.parametrize("mask_some_labels", [False, True])
def test_chunked_lm_head_loss_matches_unchunked_reference(num_chunks, mask_some_labels):
    trunk_ref, lm_head_ref, inputs, labels = _make_modules_and_data(mask_some_labels=mask_some_labels)
    trunk_chunked = copy.deepcopy(trunk_ref)
    lm_head_chunked = copy.deepcopy(lm_head_ref)

    loss_ref, grads_ref = _reference_loss_and_grads(trunk_ref, lm_head_ref, inputs, labels)
    loss_chunked, grads_chunked = _chunked_loss_and_grads(
        trunk_chunked, lm_head_chunked, inputs, labels, num_chunks
    )

    torch.testing.assert_close(loss_chunked, loss_ref)
    assert not loss_chunked.requires_grad
    for name, grad_ref in grads_ref.items():
        torch.testing.assert_close(grads_chunked[name], grad_ref, msg=lambda m: f"{name}: {m}")


def test_grad_scale_scales_gradients_but_not_loss():
    grad_scale = 0.25
    trunk_ref, lm_head_ref, inputs, labels = _make_modules_and_data()
    trunk_chunked = copy.deepcopy(trunk_ref)
    lm_head_chunked = copy.deepcopy(lm_head_ref)

    loss_ref, grads_ref = _reference_loss_and_grads(trunk_ref, lm_head_ref, inputs, labels)
    loss_chunked, grads_chunked = _chunked_loss_and_grads(
        trunk_chunked, lm_head_chunked, inputs, labels, num_chunks=4, grad_scale=grad_scale
    )

    torch.testing.assert_close(loss_chunked, loss_ref)
    for name, grad_ref in grads_ref.items():
        torch.testing.assert_close(grads_chunked[name], grad_ref * grad_scale, msg=lambda m: f"{name}: {m}")


def test_forward_only_call_matches_reference():
    trunk, lm_head, inputs, labels = _make_modules_and_data(mask_some_labels=True)
    loss_fn = ChunkedLMHeadCrossEntropyLoss(
        target_key="target", prediction_key="prediction", num_chunks=4, use_compile=False
    )
    loss_fn.bind_lm_head(TinyModelWithLMHead(lm_head))

    with torch.no_grad():
        hidden = trunk(inputs)
        result_batch = InferenceResultBatch(targets={"target": labels}, predictions={"prediction": hidden})
        loss = loss_fn(result_batch)

        logits = lm_head(hidden)
        loss_ref = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), labels.view(-1), reduction="mean", ignore_index=IGNORE_INDEX
        )

    torch.testing.assert_close(loss, loss_ref)


def test_unbound_lm_head_raises():
    loss_fn = ChunkedLMHeadCrossEntropyLoss(target_key="target", prediction_key="prediction", num_chunks=4)
    hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, requires_grad=True)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    with pytest.raises(RuntimeError, match="lm_head is not set"):
        loss_fn.compute_and_backward(hidden, labels)


def test_indivisible_sequence_length_raises():
    trunk, lm_head, inputs, labels = _make_modules_and_data()
    loss_fn = ChunkedLMHeadCrossEntropyLoss(
        target_key="target", prediction_key="prediction", num_chunks=3, use_compile=False
    )
    loss_fn.lm_head = lm_head
    hidden = trunk(inputs)
    with pytest.raises(ValueError, match="divisible"):
        loss_fn.compute_and_backward(hidden, labels)
