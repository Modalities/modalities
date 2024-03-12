from copy import deepcopy

import pytest
import torch

from modalities.models.gpt2.gpt2_model import AttentionType, CausalSelfAttention


def _get_random_input_seq(embedding_shape):
    return torch.rand(size=embedding_shape, dtype=torch.float32)


def _get_random_attention_layer(n_head_q, n_head_kv, n_embd, attention_type, block_size):
    return CausalSelfAttention(
        n_head_q=n_head_q,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        attention_type=attention_type,
        bias=False,
        dropout=0.0,
        block_size=block_size,
    )


@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd, attention_type, successful",
    [
        # Flash Attention
        (4, 4, 32, AttentionType.PYTORCH_FLASH_ATTENTION, True),
        (8, 2, 32, AttentionType.PYTORCH_FLASH_ATTENTION, True),
        (9, 8, 32, AttentionType.PYTORCH_FLASH_ATTENTION, False),
        (8, 3, 32, AttentionType.PYTORCH_FLASH_ATTENTION, False),
        # Default Attention
        (4, 4, 32, AttentionType.DEFAULT_ATTENTION, True),
        (8, 2, 32, AttentionType.DEFAULT_ATTENTION, True),
        (9, 8, 32, AttentionType.DEFAULT_ATTENTION, False),
        (8, 3, 32, AttentionType.DEFAULT_ATTENTION, False),
    ],
)
def test_forward_pass_success(n_head_q, n_head_kv, n_embd, attention_type, successful):
    batch_size = 2
    block_size = 10
    embedding_shape = (batch_size, block_size, n_embd)

    attention_layer_args = {
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "attention_type": attention_type,
        "block_size": block_size,
    }

    if not successful:
        with pytest.raises(Exception):
            _get_random_attention_layer(**attention_layer_args)
    else:
        attention_layer = _get_random_attention_layer(**attention_layer_args)
        embedded_input_seq = _get_random_input_seq(embedding_shape)
        output_tensor = attention_layer(embedded_input_seq)
        assert output_tensor.shape == embedding_shape


@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd",
    [
        (4, 4, 32),
        (8, 2, 32),
    ],
)
def test_attention_types_equality(n_head_q, n_head_kv, n_embd):
    batch_size = 2
    block_size = 10
    embedding_shape = (batch_size, block_size, n_embd)
    embedded_input_seq = _get_random_input_seq(embedding_shape)

    attention_layer_args = {
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "attention_type": AttentionType.DEFAULT_ATTENTION,
        "block_size": block_size,
    }

    attention_layer_default = _get_random_attention_layer(**attention_layer_args)
    attention_layer_flash = deepcopy(attention_layer_default)
    attention_layer_flash.flash = True

    output_tensor_default = attention_layer_default(embedded_input_seq)
    output_tensor_flash = attention_layer_flash(embedded_input_seq)
    torch.testing.assert_close(output_tensor_default, output_tensor_flash)
