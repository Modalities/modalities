import torch
from torch.nn import CrossEntropyLoss

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import MoECrossEntropyLoss
from modalities.models.moe.qwen_model import MoEBlock, QwenModel

VOCAB, NUM_EXPERTS, TOP_K, AUX_COEF, Z_COEF = 32, 4, 2, 0.01, 0.001


def _build_model() -> QwenModel:
    torch.manual_seed(0)
    model = QwenModel(
        vocab_size=VOCAB,
        max_seq_len=16,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        num_layers=2,
        moe_d_ff=24,
        moe_num_experts=NUM_EXPERTS,
        moe_top_k=TOP_K,
        moe_aux_loss_coef=AUX_COEF,
        moe_z_loss_coef=Z_COEF,
        moe_expert_backend="loop",
    )
    # expert weights are allocated uninitialized (initialized later by the model initializer in training)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return model


def _forward(model: QwenModel, seed: int) -> tuple[InferenceResultBatch, torch.Tensor, list[MoEBlock]]:
    input_ids = torch.randint(0, VOCAB, (2, 6), generator=torch.Generator().manual_seed(seed))
    logits = model(input_ids)
    targets = torch.roll(input_ids, -1, dims=1)
    batch = InferenceResultBatch(targets={"targets": targets}, predictions={"logits": logits})
    ce = CrossEntropyLoss()(logits.view(-1, VOCAB), targets.view(-1))
    blocks = [m for m in model.modules() if isinstance(m, MoEBlock)]
    return batch, ce, blocks


def _expected_moe_loss(blocks: list[MoEBlock], tokens_per_expert: torch.Tensor) -> torch.Tensor:
    f = tokens_per_expert.float() / tokens_per_expert.sum(dim=-1, keepdim=True)
    expected = torch.zeros(())
    for block, f_layer in zip(blocks, f):
        expected = expected + AUX_COEF * NUM_EXPERTS * torch.sum(f_layer * block.last_router_prob_mean)
        expected = expected + block.last_z_loss
    return expected


def test_moe_loss_without_router_statistics_matches_plain_ce():
    model = _build_model()
    loss_fn = MoECrossEntropyLoss(target_key="targets", prediction_key="logits", model=model)
    logits = torch.randn(2, 3, VOCAB)
    targets = torch.randint(0, VOCAB, (2, 3))
    batch = InferenceResultBatch(targets={"targets": targets}, predictions={"logits": logits})

    expected = CrossEntropyLoss()(logits.view(-1, VOCAB), targets.view(-1))
    assert torch.allclose(loss_fn(batch), expected)


def test_moe_loss_uses_token_counts_accumulated_over_the_global_batch():
    model = _build_model()
    loss_fn = MoECrossEntropyLoss(target_key="targets", prediction_key="logits", model=model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    # micro-batch 1: only its own token counts
    batch, ce, blocks = _forward(model, seed=1)
    counts_1 = torch.stack([b.last_tokens_per_expert for b in blocks])
    torch.testing.assert_close(loss_fn(batch), ce + _expected_moe_loss(blocks, counts_1))

    # micro-batch 2 of the same optimizer step: counts of micro-batches 1 and 2
    batch, ce, blocks = _forward(model, seed=2)
    counts_2 = torch.stack([b.last_tokens_per_expert for b in blocks])
    loss = loss_fn(batch)
    torch.testing.assert_close(loss, ce + _expected_moe_loss(blocks, counts_1 + counts_2))

    # the load balancing loss is differentiable w.r.t. the router
    loss.backward()
    assert all(b.router.weight.grad is not None and b.router.weight.grad.abs().sum() > 0 for b in blocks)

    # an optimizer step starts a new global batch
    optimizer.step()
    batch, ce, blocks = _forward(model, seed=3)
    counts_3 = torch.stack([b.last_tokens_per_expert for b in blocks])
    torch.testing.assert_close(loss_fn(batch), ce + _expected_moe_loss(blocks, counts_3))


def test_moe_loss_in_evaluation_uses_local_counts_and_keeps_training_state():
    model = _build_model()
    loss_fn = MoECrossEntropyLoss(target_key="targets", prediction_key="logits", model=model)

    batch, _, blocks = _forward(model, seed=1)
    counts_train = torch.stack([b.last_tokens_per_expert for b in blocks])
    loss_fn(batch)

    with torch.no_grad():
        batch, ce, blocks = _forward(model, seed=2)
        counts_eval = torch.stack([b.last_tokens_per_expert for b in blocks])
        torch.testing.assert_close(loss_fn(batch), ce + _expected_moe_loss(blocks, counts_eval))

    # evaluation did not modify the accumulated training statistics
    batch, ce, blocks = _forward(model, seed=3)
    counts_3 = torch.stack([b.last_tokens_per_expert for b in blocks])
    torch.testing.assert_close(loss_fn(batch), ce + _expected_moe_loss(blocks, counts_train + counts_3))
