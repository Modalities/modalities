import pytest
import torch

from modalities.models.gpt2.gpt2_model import AttentionType, CausalSelfAttention


def _get_random_input_seq(embedding_shape):
    flash_attn_supported_dtype = torch.bfloat16
    return torch.rand(size=embedding_shape, dtype=flash_attn_supported_dtype)


def _get_random_attention_layer(n_head_q, n_head_kv, n_embd, attention_type, block_size):
    self_attention_layer = CausalSelfAttention(
        n_head_q=n_head_q,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        attention_type=attention_type,
        bias=False,
        dropout=0.0,
        block_size=block_size,
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

    attention_layer_args = {
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "attention_type": AttentionType.DEFAULT_ATTENTION,
        "block_size": block_size,
    }

    if not successful:
        with pytest.raises(Exception):
            _get_random_attention_layer(**attention_layer_args)
    else:
        attention_layer = _get_random_attention_layer(**attention_layer_args).cuda()
        embedded_input_seq = _get_random_input_seq(embedding_shape).cuda()
        output_tensor = attention_layer(embedded_input_seq)
        assert output_tensor.shape == embedding_shape
