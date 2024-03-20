import pytest
import torch
import torch.nn.functional as F
from einops import einsum, rearrange

from modalities.models.gpt2.gpt2_model import AttentionConfig, CausalSelfAttention


def _get_random_input_seq(embedding_shape):
    flash_attn_supported_dtype = torch.bfloat16
    return torch.rand(size=embedding_shape, dtype=flash_attn_supported_dtype)


def _get_random_attention_layer(n_head_q, n_head_kv, n_embd, block_size, attention_config):
    self_attention_layer = CausalSelfAttention(
        n_head_q=n_head_q,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        bias=False,
        dropout=0.0,
        block_size=block_size,
        attention_config=attention_config,
    ).cuda()
    self_attention_layer.q_attn = self_attention_layer.q_attn.bfloat16()
    self_attention_layer.k_attn = self_attention_layer.k_attn.bfloat16()
    self_attention_layer.v_attn = self_attention_layer.v_attn.bfloat16()
    self_attention_layer.c_proj = self_attention_layer.c_proj.bfloat16()
    return self_attention_layer


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd, successful",
    [
        (4, 4, 32, True),
        (8, 2, 32, True),
        (9, 8, 32, False),
        (8, 3, 32, False),
    ],
)
def test_forward_pass_success(n_head_q, n_head_kv, n_embd, successful):
    batch_size = 2
    block_size = 10
    embedding_shape = (batch_size, block_size, n_embd)
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer_args = {
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "block_size": block_size,
        "attention_config": attention_config,
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
def test_forward_equality():
    # Source: https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a

    # shapes: (batch_size, seq_len, num_heads, head_dim)
    query_orig = torch.rand(1, 12, 8, 16, dtype=torch.bfloat16).cuda()
    key_orig = torch.rand(1, 12, 2, 16, dtype=torch.bfloat16).cuda()
    value_orig = torch.rand(1, 12, 2, 16, dtype=torch.bfloat16).cuda()

    # define number of heads in one group, in this toy example we have 2 kv_heads,
    # so this means we will have 2 groups of size 4 each
    num_head_groups = query_orig.shape[2] // key_orig.shape[2]
    scale = query_orig.size(-1) ** 0.5

    # Swap seq len with num_heads to accelerate computations
    query = rearrange(query_orig, "b s hq d -> b hq s d")
    key = rearrange(key_orig, "b s hk d -> b hk s d")
    value = rearrange(value_orig, "b s hv d -> b hv s d")

    # split query num heads in groups by introducing additional 'g' dimension
    query = rearrange(query, "b (h g) s d -> b g h s d", g=num_head_groups)

    # calculate the attention scores and sum over the group dim to perform averaging
    scores = einsum(query, key, "b g h s d, b hk s d -> b g h s d").reshape(1, 12, 8, 16)
    attention = F.softmax(scores / scale, dim=-1)

    # apply weights to the value head
    out = einsum(attention, value, "b s h d, b hk s d -> b h s d")

    # reshape back to original dimensions
    out = rearrange(out, "b h n d -> b n h d")

    # FlasAttention
    # (B, nh_q, T, hs)
    query_orig = query_orig.transpose(1, 2)
    key_orig = key_orig.transpose(1, 2)
    value_orig = value_orig.transpose(1, 2)
    out_flash = CausalSelfAttention.execute_flash_attention(query_orig, key_orig, value_orig, dropout=0.0)

    assert torch.equal(out, out_flash)
