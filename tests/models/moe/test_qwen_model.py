import pytest
import torch

from modalities.models.moe.qwen_model import GroupedExperts, QwenModel


def _build_tiny_qwen_model() -> QwenModel:
    model = QwenModel(
        vocab_size=32,
        max_seq_len=16,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        num_layers=1,
        moe_d_ff=24,
        moe_num_experts=4,
        moe_top_k=2,
        moe_min_capacity=1,
        moe_overflow_policy="residual",
        moe_aux_loss_coef=0.01,
        moe_z_loss_coef=0.0,
    )
    # expert weights are allocated uninitialized (initialized later by the model initializer in training)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return model


def test_qwen_model_forward_dict_output_shape():
    torch.manual_seed(0)
    model = _build_tiny_qwen_model()
    batch_size, seq_len = 2, 5

    input_ids = torch.randint(0, 32, (batch_size, seq_len), dtype=torch.long)
    output = model({"input_ids": input_ids})

    assert "logits" in output
    assert output["logits"].shape == (batch_size, seq_len, 32)


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
def test_grouped_experts_forward_preserves_input_dtype(backend: str):
    # grouped_mm requires 16-byte aligned strides (d_ff * 2 bytes for bf16)
    experts = GroupedExperts(num_experts=2, d_model=8, d_ff=16, ffn_dropout=0.0, backend=backend)
    experts.reset_parameters()

    # Input in bf16 while expert weights are initialized in fp32.
    routed_input = torch.randn(4, 8, dtype=torch.bfloat16)
    num_tokens_per_expert = torch.tensor([3, 1], dtype=torch.long)

    out = experts(routed_input, num_tokens_per_expert)

    assert out.shape == routed_input.shape
    assert out.dtype == routed_input.dtype


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
def test_grouped_experts_matches_per_expert_reference(backend: str):
    torch.manual_seed(2)
    experts = GroupedExperts(num_experts=3, d_model=8, d_ff=12, ffn_dropout=0.0, backend=backend)
    experts.reset_parameters()
    counts = [4, 0, 3]
    routed_input = torch.randn(sum(counts), 8)

    out = experts(routed_input, torch.tensor(counts))

    expected = []
    for e, x in enumerate(torch.split(routed_input, counts)):
        h = torch.nn.functional.silu(x @ experts.w1[e].T) * (x @ experts.w2[e].T)
        expected.append(h @ experts.w3[e].T)
    torch.testing.assert_close(out, torch.cat(expected))


def test_grouped_experts_zeroes_dropped_rows():
    experts = GroupedExperts(num_experts=2, d_model=8, d_ff=12, ffn_dropout=0.0, backend="loop")
    experts.reset_parameters()
    # 5 rows, but only 3 are assigned to experts; the trailing 2 rows were dropped by a capacity limit
    out = experts(torch.randn(5, 8), torch.tensor([2, 1]))
    assert out.shape == (5, 8)
    assert torch.all(out[3:] == 0)


def test_grouped_experts_init_std_is_per_expert():
    experts = GroupedExperts(num_experts=4, d_model=256, d_ff=128, ffn_dropout=0.0)
    experts.reset_parameters()
    for w in (experts.w1, experts.w2, experts.w3):
        assert 0.015 < w.std().item() < 0.025


def test_moe_block_records_router_statistics_after_forward():
    torch.manual_seed(1)
    model = _build_tiny_qwen_model()
    input_ids = torch.randint(0, 32, (2, 4), dtype=torch.long)

    _ = model({"input_ids": input_ids})

    moe_block = next(iter(model.layers.values())).ffn
    # local token counts per expert (top_k assignments per token) and the mean routing probabilities
    assert moe_block.last_tokens_per_expert.sum().item() == 2 * 4 * moe_block.top_k
    assert moe_block.last_router_prob_mean.requires_grad
    torch.testing.assert_close(moe_block.last_router_prob_mean.sum(), torch.tensor(1.0))
