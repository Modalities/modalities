import pytest
import torch

from modalities.models.gpt2.gpt2_model import AttentionType, CausalSelfAttention


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
    embedded_input_seq = torch.rand(size=embedding_shape, dtype=torch.float32)

    def forward_pass(n_head_q, n_head_kv, n_embd, attention_type, block_size, embedded_input_seq):
        attention_layer = CausalSelfAttention(
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            attention_type=attention_type,
            bias=False,
            dropout=0.0,
            block_size=block_size,
        )
        output_tensor: torch.Tensor = attention_layer(embedded_input_seq)
        return output_tensor

    if not successful:
        with pytest.raises(Exception):
            forward_pass(n_head_q, n_head_kv, n_embd, attention_type, block_size, embedded_input_seq)
    else:
        output_tensor = forward_pass(n_head_q, n_head_kv, n_embd, attention_type, block_size, embedded_input_seq)
        assert output_tensor.shape == embedding_shape
