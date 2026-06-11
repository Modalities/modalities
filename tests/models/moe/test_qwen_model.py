import torch

from modalities.models.moe.qwen_model import GroupedExperts, QwenModel


def _build_tiny_qwen_model() -> QwenModel:
    return QwenModel(
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
        moe_capacity_factor=1.25,
        moe_min_capacity=1,
        moe_overflow_policy="residual",
        moe_aux_loss_coef=0.01,
        moe_z_loss_coef=0.0,
    )


def test_qwen_model_forward_dict_output_shape():
    torch.manual_seed(0)
    model = _build_tiny_qwen_model()
    batch_size, seq_len = 2, 5

    input_ids = torch.randint(0, 32, (batch_size, seq_len), dtype=torch.long)
    output = model({"input_ids": input_ids})

    assert "logits" in output
    assert output["logits"].shape == (batch_size, seq_len, 32)


def test_grouped_experts_forward_local_preserves_input_dtype():
    experts = GroupedExperts(num_experts=2, d_model=8, d_ff=12, ffn_dropout=0.0)
    experts.reset_parameters()

    # Input in bf16 while expert weights are initialized in fp32.
    routed_input = torch.randn(4, 8, dtype=torch.bfloat16)
    num_tokens_per_expert = torch.tensor([2, 2], dtype=torch.long)

    out = experts._forward_local(routed_input=routed_input, num_tokens_per_expert=num_tokens_per_expert)

    assert out.shape == routed_input.shape
    assert out.dtype == routed_input.dtype


def test_transformer_block_exposes_aux_loss_after_forward():
    torch.manual_seed(1)
    model = _build_tiny_qwen_model()
    input_ids = torch.randint(0, 32, (2, 4), dtype=torch.long)

    _ = model({"input_ids": input_ids})

    first_layer = next(iter(model.layers.values()))
    assert first_layer.aux_loss is not None
