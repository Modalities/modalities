import pytest
import torch

from modalities.models.gpt2.gpt2_model import AttentionType, CausalSelfAttention


@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd, attention_type, successful",
    [
        # TODO: Flash Atttention
        #        (4, 4, 32, AttentionType.DEFAULT_ATTENTION, True),
        (8, 2, 32, AttentionType.DEFAULT_ATTENTION, True),
        #        (9, 8, 32, AttentionType.DEFAULT_ATTENTION, False),
        #        (8, 3, 32, AttentionType.DEFAULT_ATTENTION, False),
    ],
)
def test_grouped_query_attention_forward(n_head_q, n_head_kv, n_embd, attention_type, successful):
    batch_size = 2
    block_size = 10
    embedding_shape = (batch_size, block_size, n_embd)
    embedded_input_seq = torch.rand(size=embedding_shape, dtype=torch.float32)

    def attention_forward_pass(attention_type, block_size, embedded_input_seq, n_embd, n_head_kv, n_head_q):
        attention_layer = CausalSelfAttention(
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            attention_type=attention_type,
            bias=False,
            dropout=False,
            block_size=block_size,
        )
        output_tensor: torch.Tensor = attention_layer(embedded_input_seq)
        return output_tensor

    if not successful:
        with pytest.raises(Exception):
            attention_forward_pass(attention_type, block_size, embedded_input_seq, n_embd, n_head_kv, n_head_q)
    else:
        output_tensor = attention_forward_pass(
            attention_type, block_size, embedded_input_seq, n_embd, n_head_kv, n_head_q
        )
        assert output_tensor.size() == embedding_shape
