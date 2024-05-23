import pytest
import torch

from modalities.models.gpt2.gpt2_model import AttentionConfig, CausalSelfAttention


def _get_random_input_seq(embedding_shape):
    flash_attn_supported_dtype = torch.bfloat16
    return torch.rand(size=embedding_shape, dtype=flash_attn_supported_dtype)


def _get_random_attention_layer(n_head_q, n_head_kv, n_embd, block_size, attention_impl, attention_config):
    self_attention_layer = CausalSelfAttention(
        n_head_q=n_head_q,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        bias=False,
        dropout=0.0,
        block_size=block_size,
        attention_config=attention_config,
        attention_impl=attention_impl,
    ).cuda()
    self_attention_layer.q_attn = self_attention_layer.q_attn.bfloat16()
    self_attention_layer.k_attn = self_attention_layer.k_attn.bfloat16()
    self_attention_layer.v_attn = self_attention_layer.v_attn.bfloat16()
    self_attention_layer.c_proj = self_attention_layer.c_proj.bfloat16()
    return self_attention_layer


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd, attention_impl, successful",
    [
        # dao_flash
        (4, 4, 32, "dao_flash", True),
        (8, 2, 32, "dao_flash", True),
        (9, 8, 32, "dao_flash", False),
        (8, 3, 32, "dao_flash", False),
        # pytorch_flash
        (4, 4, 32, "pytorch_flash", True),
        (8, 2, 32, "pytorch_flash", False),
        (9, 8, 32, "pytorch_flash", False),
        (8, 3, 32, "pytorch_flash", False),
        # manual
        (4, 4, 32, "manual", True),
        (8, 2, 32, "manual", False),
        (9, 8, 32, "manual", False),
        (8, 3, 32, "manual", False),
    ],
)
def test_forward_pass_success(n_head_q, n_head_kv, n_embd, attention_impl, successful):
    batch_size = 2
    block_size = 10
    embedding_shape = (batch_size, block_size - 1, n_embd)
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer_args = {
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "block_size": block_size,
        "attention_config": attention_config,
        "attention_impl": attention_impl,
    }

    if not successful:
        with pytest.raises(Exception):
            _get_random_attention_layer(**attention_layer_args)
    else:
        attention_layer = _get_random_attention_layer(**attention_layer_args).cuda()
        embedded_input_seq = _get_random_input_seq(embedding_shape).cuda()
        output_tensor = attention_layer(embedded_input_seq)
        assert output_tensor.shape == embedding_shape


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
@pytest.mark.parametrize(
    "seq_length, n_head_q, n_head_kv, head_dim, attention_impl",
    [
        # dao_flash
        (12, 4, 4, 32, "dao_flash"),
        (12, 8, 2, 32, "dao_flash"),
        (16, 8, 8, 16, "dao_flash"),
        # pytorch_flash
        (12, 4, 4, 32, "pytorch_flash"),
        (16, 8, 8, 16, "pytorch_flash"),
        # manual
        (12, 4, 4, 32, "manual"),
        (16, 8, 8, 16, "manual"),
    ],
)
def test_forward_pass_shapes(seq_length, n_head_q, n_head_kv, head_dim, attention_impl):
    # Source: https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a

    # shapes: (batch_size, num_heads, seq_length, head_dim)
    query_orig = torch.rand(2, n_head_q, seq_length, head_dim, dtype=torch.bfloat16).cuda()
    key_orig = torch.rand(2, n_head_kv, seq_length, head_dim, dtype=torch.bfloat16).cuda()
    value_orig = torch.rand(2, n_head_kv, seq_length, head_dim, dtype=torch.bfloat16).cuda()

    attn_mask = (
        torch.tril(torch.ones(seq_length, seq_length)).view(1, 1, seq_length, seq_length).to("cuda")
        if attention_impl == "manual"
        else None
    )
    out_flash = CausalSelfAttention.execute_flash_attention(
        query_orig,
        key_orig,
        value_orig,
        dropout=0.0,
        attention_impl=attention_impl,
        attn_mask=attn_mask,
    )

    # shape: (batch_size, seq_length, num_heads, head_dim)
    assert out_flash.shape == (2, seq_length, n_head_q, head_dim)
