import pytest
import torch

from modalities.models.components.moe.experts import ExpertsBackend, GroupedExperts, squared_relu
from modalities.models.components.moe.moe import MoE
from modalities.models.components.moe.router import RouterScoreFunction, TopKRouter
from modalities.models.nemotron.nemotron_mlp import SquaredReLUMLP

N_EMBD = 16
NUM_EXPERTS = 8
TOP_K = 2
# Both matmul inner dimensions must be multiples of 8 so that the grouped_mm backend is usable.
FFN_HIDDEN = 24


def _make_router(**overrides) -> TopKRouter:
    torch.manual_seed(0)
    kwargs = dict(n_embd=N_EMBD, num_experts=NUM_EXPERTS, top_k=TOP_K)
    return TopKRouter(**{**kwargs, **overrides})


def _make_experts(**overrides) -> GroupedExperts:
    torch.manual_seed(0)
    kwargs = dict(n_embd=N_EMBD, ffn_hidden=FFN_HIDDEN, num_experts=NUM_EXPERTS, backend=ExpertsBackend.LOOPED)
    experts = GroupedExperts(**{**kwargs, **overrides})
    with torch.no_grad():
        experts.w1.normal_(0, 0.02)
        experts.w2.normal_(0, 0.02)
    return experts


def _make_moe(aux_loss_coeff: float = 0.0, with_shared: bool = False) -> MoE:
    shared = SquaredReLUMLP(n_embd=N_EMBD, ffn_hidden=2 * FFN_HIDDEN) if with_shared else None
    return MoE(
        router=_make_router(),
        experts=_make_experts(),
        shared_experts=shared,
        aux_loss_coeff=aux_loss_coeff,
    )


# --------------------------------------------------------------------------------------------
# Activation
# --------------------------------------------------------------------------------------------


def test_squared_relu_zeroes_negatives_and_squares_positives():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
    torch.testing.assert_close(squared_relu(x), torch.tensor([0.0, 0.0, 0.0, 0.25, 9.0]))


# --------------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------------


def test_router_output_shapes_and_dtypes():
    router = _make_router()
    x = torch.randn(10, N_EMBD)
    weights, indices, scores = router(x)

    assert weights.shape == (10, TOP_K)
    assert indices.shape == (10, TOP_K)
    assert scores.shape == (10, NUM_EXPERTS)
    assert weights.dtype == x.dtype
    assert indices.dtype == torch.int64
    # Scores stay in the router dtype so the auxiliary loss is computed in fp32.
    assert scores.dtype == torch.float32


def test_router_selects_distinct_experts_per_token():
    router = _make_router()
    _, indices, _ = router(torch.randn(32, N_EMBD))
    for row in indices:
        assert len(set(row.tolist())) == TOP_K


def test_sigmoid_router_weights_sum_to_route_scale():
    router = _make_router(route_scale=2.5)
    weights, _, _ = router(torch.randn(20, N_EMBD))
    # Renormalization over the selected experts followed by the constant rescaling.
    torch.testing.assert_close(weights.sum(dim=-1), torch.full((20,), 2.5), rtol=1e-5, atol=1e-5)


def test_sigmoid_router_with_top_k_one_does_not_renormalize():
    # With top_k == 1 the reference implementation keeps the raw sigmoid score rather than
    # normalizing it to 1, so the weight must stay below 1.
    router = _make_router(top_k=1, route_scale=1.0)
    weights, indices, scores = router(torch.randn(20, N_EMBD))
    torch.testing.assert_close(weights.squeeze(-1), torch.gather(scores, 1, indices).squeeze(-1))
    assert (weights < 1.0).all()


def test_softmax_router_weights_form_a_distribution():
    router = _make_router(score_function=RouterScoreFunction.SOFTMAX)
    weights, _, scores = router(torch.randn(20, N_EMBD))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(20), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(20), rtol=1e-5, atol=1e-5)


def test_expert_bias_changes_selection_but_not_returned_weights():
    # The load-balancing bias must steer which experts are picked without leaking into the
    # forward signal; otherwise balancing would distort the model's output.
    router = _make_router(route_scale=1.0)
    x = torch.randn(64, N_EMBD)
    _, baseline_indices, scores = router(x)

    starved_expert = 5
    with torch.no_grad():
        router.expert_bias[starved_expert] = 100.0
    weights, indices, _ = router(x)

    assert (indices == starved_expert).any()
    assert not torch.equal(indices, baseline_indices)
    # Weights are still the unbiased sigmoid scores of the selected experts (renormalized).
    selected = torch.gather(scores, 1, indices)
    torch.testing.assert_close(weights, selected / selected.sum(dim=-1, keepdim=True), rtol=1e-5, atol=1e-6)


def test_router_without_expert_bias_has_no_bias_buffer():
    router = _make_router(use_expert_bias=False)
    assert router.expert_bias is None
    assert "expert_bias" not in dict(router.named_buffers())


def test_expert_bias_is_persistent_and_token_counter_is_not():
    router = _make_router()
    state_dict = router.state_dict()
    assert "expert_bias" in state_dict
    assert "tokens_per_expert" not in state_dict


def test_router_rejects_invalid_top_k():
    with pytest.raises(ValueError, match=r"top_k must be in"):
        _make_router(top_k=0)
    with pytest.raises(ValueError, match=r"top_k must be in"):
        _make_router(top_k=NUM_EXPERTS + 1)


def test_router_reset_parameters_zeroes_load_balancing_state():
    router = _make_router()
    with torch.no_grad():
        router.expert_bias.fill_(3.0)
        router.tokens_per_expert.fill_(7.0)
    router.reset_parameters()
    torch.testing.assert_close(router.expert_bias, torch.zeros(NUM_EXPERTS))
    torch.testing.assert_close(router.tokens_per_expert, torch.zeros(NUM_EXPERTS))


# --------------------------------------------------------------------------------------------
# Grouped experts
# --------------------------------------------------------------------------------------------


def test_grouped_experts_parameter_shapes_are_non_gated():
    experts = _make_experts()
    # Two matrices per expert (no gate projection), expert dimension first for FSDP sharding.
    assert experts.w1.shape == (NUM_EXPERTS, FFN_HIDDEN, N_EMBD)
    assert experts.w2.shape == (NUM_EXPERTS, N_EMBD, FFN_HIDDEN)
    assert len(list(experts.parameters())) == 2


def test_looped_experts_apply_the_right_expert_to_each_slice():
    experts = _make_experts()
    tokens_per_expert = torch.tensor([2, 0, 3, 1, 0, 0, 0, 0])
    x_sorted = torch.randn(int(tokens_per_expert.sum()), N_EMBD)
    out = experts(x_sorted, tokens_per_expert)

    start = 0
    for expert_idx, count in enumerate(tokens_per_expert.tolist()):
        if count == 0:
            continue
        chunk = x_sorted[start : start + count]
        expected = squared_relu(chunk @ experts.w1[expert_idx].T) @ experts.w2[expert_idx].T
        torch.testing.assert_close(out[start : start + count], expected, rtol=1e-5, atol=1e-6)
        start += count


def test_experts_handle_all_tokens_on_one_expert():
    experts = _make_experts()
    tokens_per_expert = torch.zeros(NUM_EXPERTS, dtype=torch.int64)
    tokens_per_expert[3] = 5
    out = experts(torch.randn(5, N_EMBD), tokens_per_expert)
    assert out.shape == (5, N_EMBD)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch._grouped_mm requires CUDA")
def test_grouped_mm_backend_matches_looped_backend():
    torch.manual_seed(0)
    looped = _make_experts(backend=ExpertsBackend.LOOPED).cuda().bfloat16()
    grouped = _make_experts(backend=ExpertsBackend.GROUPED_MM).cuda().bfloat16()
    grouped.load_state_dict(looped.state_dict())

    tokens_per_expert = torch.tensor([4, 2, 0, 6, 1, 3, 0, 8], device="cuda")
    x_sorted = torch.randn(int(tokens_per_expert.sum()), N_EMBD, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        grouped(x_sorted, tokens_per_expert), looped(x_sorted, tokens_per_expert), rtol=2e-2, atol=2e-2
    )


def test_grouped_mm_backend_falls_back_on_cpu():
    # The backend selection must degrade gracefully so that CPU tests and CPU inference work.
    experts = _make_experts(backend=ExpertsBackend.GROUPED_MM)
    tokens_per_expert = torch.tensor([2, 1, 0, 0, 0, 0, 0, 1])
    out = experts(torch.randn(4, N_EMBD), tokens_per_expert)
    assert out.shape == (4, N_EMBD)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n_embd,ffn_hidden", [(12, 24), (16, 20), (12, 12)])
def test_grouped_mm_backend_rejects_unaligned_dimensions(n_embd, ffn_hidden):
    # torch._grouped_mm needs 16-byte aligned strides. Catching that at construction time is far
    # more useful than a "strides should be multiple of 16 bytes" RuntimeError mid-training.
    with pytest.raises(ValueError, match="must be a multiple of 8"):
        GroupedExperts(n_embd=n_embd, ffn_hidden=ffn_hidden, num_experts=NUM_EXPERTS, backend=ExpertsBackend.GROUPED_MM)


def test_looped_backend_accepts_unaligned_dimensions():
    experts = GroupedExperts(n_embd=12, ffn_hidden=12, num_experts=2, backend=ExpertsBackend.LOOPED)
    with torch.no_grad():
        experts.w1.normal_(0, 0.02)
        experts.w2.normal_(0, 0.02)
    out = experts(torch.randn(3, 12), torch.tensor([1, 2]))
    assert out.shape == (3, 12)


# --------------------------------------------------------------------------------------------
# MoE layer
# --------------------------------------------------------------------------------------------


def test_moe_forward_preserves_shape():
    moe = _make_moe()
    x = torch.randn(2, 6, N_EMBD)
    out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_moe_matches_an_explicit_per_token_reference():
    # The dispatch/sort/combine machinery is the easiest place to introduce an indexing bug, so
    # compare against a naive loop over tokens.
    moe = _make_moe()
    x = torch.randn(2, 5, N_EMBD)
    out = moe(x)

    x_flat = x.reshape(-1, N_EMBD)
    weights, indices, _ = moe.router(x_flat)
    expected = torch.zeros_like(x_flat)
    for token_idx in range(x_flat.shape[0]):
        for slot in range(TOP_K):
            expert_idx = indices[token_idx, slot]
            hidden = squared_relu(x_flat[token_idx] @ moe.experts.w1[expert_idx].T)
            expected[token_idx] += weights[token_idx, slot] * (hidden @ moe.experts.w2[expert_idx].T)

    torch.testing.assert_close(out.reshape(-1, N_EMBD), expected, rtol=1e-4, atol=1e-5)


def test_moe_shared_expert_output_is_added():
    routed_only = _make_moe(with_shared=False)
    with_shared = _make_moe(with_shared=True)
    with_shared.router.load_state_dict(routed_only.router.state_dict())
    with_shared.experts.load_state_dict(routed_only.experts.state_dict())

    x = torch.randn(2, 4, N_EMBD)
    expected = routed_only(x) + with_shared.shared_experts(x)
    torch.testing.assert_close(with_shared(x), expected, rtol=1e-5, atol=1e-6)


def test_moe_counts_tokens_per_expert():
    moe = _make_moe()
    x = torch.randn(2, 8, N_EMBD)
    moe(x)
    # Every token contributes exactly top_k routed slots.
    assert moe.router.tokens_per_expert.sum().item() == 2 * 8 * TOP_K


def test_moe_token_counts_accumulate_across_micro_batches():
    # Gradient accumulation means several forwards per optimizer step; the counter must sum them.
    moe = _make_moe()
    moe(torch.randn(1, 4, N_EMBD))
    after_first = moe.router.tokens_per_expert.clone()
    moe(torch.randn(1, 4, N_EMBD))
    assert moe.router.tokens_per_expert.sum() == 2 * after_first.sum()


def test_moe_aux_loss_is_none_when_disabled():
    moe = _make_moe(aux_loss_coeff=0.0)
    moe(torch.randn(2, 4, N_EMBD))
    assert moe.last_aux_loss is None


def test_moe_aux_loss_is_a_scalar_in_the_autograd_graph():
    moe = _make_moe(aux_loss_coeff=1e-2)
    moe(torch.randn(2, 4, N_EMBD))
    assert moe.last_aux_loss.ndim == 0
    assert moe.last_aux_loss.requires_grad
    moe.last_aux_loss.backward()
    assert moe.router.gate.weight.grad is not None


def test_moe_aux_loss_is_overwritten_not_accumulated():
    # Overwriting is what makes the loss correct under activation checkpointing, where the
    # forward pass is replayed during the backward pass.
    moe = _make_moe(aux_loss_coeff=1e-2)
    x = torch.randn(2, 4, N_EMBD)
    moe(x)
    first = moe.last_aux_loss.item()
    moe(x)
    assert moe.last_aux_loss.item() == pytest.approx(first)


def test_moe_aux_loss_is_minimal_for_perfectly_balanced_routing():
    # The Switch loss is E * <f, P> with sum(f) == sum(P) == 1. It attains its floor of 1.0 when
    # both the load f and the router probability P are uniform, and grows as the two become
    # correlated (i.e. the router keeps preferring the experts that are already overloaded).
    moe = _make_moe(aux_loss_coeff=1.0)
    num_tokens = 64
    uniform_scores = torch.full((num_tokens, NUM_EXPERTS), 1.0 / NUM_EXPERTS)
    # Round-robin assignment gives every expert exactly the same number of slots.
    balanced_indices = torch.stack(
        [torch.arange(num_tokens) % NUM_EXPERTS, (torch.arange(num_tokens) + 1) % NUM_EXPERTS], dim=-1
    )
    balanced = moe._compute_aux_loss(scores=uniform_scores, top_indices=balanced_indices, batch_size=1)
    torch.testing.assert_close(balanced, torch.tensor(1.0), rtol=1e-5, atol=1e-6)

    # Collapse: the router puts all its probability mass on experts 0 and 1 and also routes
    # every token there, which is the pathology the loss is meant to punish.
    peaked_scores = torch.full((num_tokens, NUM_EXPERTS), 1e-6)
    peaked_scores[:, 0] = 0.5
    peaked_scores[:, 1] = 0.5
    collapsed_indices = torch.zeros_like(balanced_indices)
    collapsed_indices[:, 1] = 1
    collapsed = moe._compute_aux_loss(scores=peaked_scores, top_indices=collapsed_indices, batch_size=1)
    assert collapsed > balanced


def test_moe_aux_loss_is_invariant_to_the_load_when_probabilities_are_uniform():
    # A property worth pinning down explicitly: with uniform P the loss cannot distinguish loads,
    # because sum(f) is always 1. The gradient therefore acts on the router probabilities, not on
    # the (non-differentiable) assignment counts.
    moe = _make_moe(aux_loss_coeff=1.0)
    num_tokens = 32
    uniform_scores = torch.full((num_tokens, NUM_EXPERTS), 1.0 / NUM_EXPERTS)
    balanced_indices = torch.stack(
        [torch.arange(num_tokens) % NUM_EXPERTS, (torch.arange(num_tokens) + 1) % NUM_EXPERTS], dim=-1
    )
    collapsed_indices = torch.zeros_like(balanced_indices)
    collapsed_indices[:, 1] = 1

    balanced = moe._compute_aux_loss(scores=uniform_scores, top_indices=balanced_indices, batch_size=1)
    collapsed = moe._compute_aux_loss(scores=uniform_scores, top_indices=collapsed_indices, batch_size=1)
    torch.testing.assert_close(balanced, collapsed, rtol=1e-6, atol=1e-7)


def test_moe_aux_loss_scales_with_the_coefficient():
    x = torch.randn(2, 4, N_EMBD)
    small, large = _make_moe(aux_loss_coeff=1e-3), _make_moe(aux_loss_coeff=1e-2)
    small(x)
    large(x)
    torch.testing.assert_close(large.last_aux_loss, small.last_aux_loss * 10.0, rtol=1e-4, atol=1e-8)


def test_moe_aux_loss_is_computed_per_sequence():
    # Each sequence collapses onto its own pair of experts. Pooled over the batch this looks
    # balanced (four experts used evenly), so a batch-level loss barely reacts. The per-sequence
    # loss sees each collapse in isolation and penalizes it. That difference is precisely why
    # Nemotron uses the sequence-level variant.
    moe = _make_moe(aux_loss_coeff=1.0)
    num_tokens_per_seq = 16
    scores = torch.full((2 * num_tokens_per_seq, NUM_EXPERTS), 1e-6)
    scores[:num_tokens_per_seq, 0] = 0.5
    scores[:num_tokens_per_seq, 1] = 0.5
    scores[num_tokens_per_seq:, 2] = 0.5
    scores[num_tokens_per_seq:, 3] = 0.5

    indices = torch.zeros(2 * num_tokens_per_seq, TOP_K, dtype=torch.int64)
    indices[:num_tokens_per_seq] = torch.tensor([0, 1])
    indices[num_tokens_per_seq:] = torch.tensor([2, 3])

    per_sequence = moe._compute_aux_loss(scores=scores, top_indices=indices, batch_size=2)
    pooled = moe._compute_aux_loss(scores=scores, top_indices=indices, batch_size=1)
    assert per_sequence > pooled
    # Sanity: two experts out of eight carrying a whole sequence is a 4x imbalance.
    torch.testing.assert_close(per_sequence, torch.tensor(4.0), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(pooled, torch.tensor(2.0), rtol=1e-3, atol=1e-3)


def test_moe_is_differentiable_through_router_and_experts():
    moe = _make_moe(with_shared=True)
    x = torch.randn(2, 6, N_EMBD, requires_grad=True)
    moe(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert moe.router.gate.weight.grad is not None
    assert moe.experts.w1.grad is not None
    assert moe.experts.w2.grad is not None
    assert moe.shared_experts.c_fc.weight.grad is not None


def test_moe_gradients_only_reach_selected_experts():
    # Sparsity is the whole point: an expert that received no token must receive no gradient.
    moe = _make_moe()
    x = torch.randn(1, 2, N_EMBD)
    _, indices, _ = moe.router(x.reshape(-1, N_EMBD))
    selected = set(indices.reshape(-1).tolist())
    unselected = sorted(set(range(NUM_EXPERTS)) - selected)
    assert unselected, "test requires at least one unrouted expert"

    moe(x).sum().backward()
    for expert_idx in unselected:
        torch.testing.assert_close(moe.experts.w1.grad[expert_idx], torch.zeros_like(moe.experts.w1[expert_idx]))


def test_moe_rejects_mismatched_expert_counts():
    with pytest.raises(ValueError, match="disagree on the number of experts"):
        MoE(router=_make_router(), experts=_make_experts(num_experts=NUM_EXPERTS + 1))


def test_moe_rejects_negative_aux_loss_coeff():
    with pytest.raises(ValueError, match="must be non-negative"):
        MoE(router=_make_router(), experts=_make_experts(), aux_loss_coeff=-1.0)


def test_nemotron_3_nano_moe_dimensions():
    # Model report Table 1: 128 routed experts of dim 1856, 2 shared experts, top-6 routing.
    # Megatron realizes the two shared experts as a single MLP of twice the expert dimension.
    moe = MoE(
        router=TopKRouter(n_embd=2688, num_experts=128, top_k=6, route_scale=2.5),
        experts=GroupedExperts(n_embd=2688, ffn_hidden=1856, num_experts=128, backend=ExpertsBackend.LOOPED),
        shared_experts=SquaredReLUMLP(n_embd=2688, ffn_hidden=2 * 1856),
        aux_loss_coeff=1e-4,
    )
    routed_params = sum(p.numel() for p in moe.experts.parameters())
    assert routed_params == 128 * 2 * 1856 * 2688
    # Active routed parameters per token: 6 of 128 experts.
    assert routed_params * 6 // 128 == 6 * 2 * 1856 * 2688
    shared_params = sum(p.numel() for p in moe.shared_experts.parameters())
    assert shared_params == 2 * 3712 * 2688
