import pytest
import torch

from modalities.nn.attention import AttentionType, MultiHeadAttention


@pytest.mark.parametrize(
    "attention_type", [AttentionType.CAUSAL_SELF_ATTENTION, AttentionType.NON_CAUSAL_SELF_ATTENTION]
)
def test_attention_forward(attention_type):
    model = MultiHeadAttention(n_embd=64, n_head=8, attention_type=attention_type)
    dummy_input = torch.randn(1, 256, 64)
    out = model(dummy_input)
    assert out.shape == (1, 256, 64)


def test_attention_with_cross_attention_forward():
    model = MultiHeadAttention(n_embd=64, n_head=8, attention_type=AttentionType.CROSS_ATTENTION)
    dummy_input = torch.randn(1, 256, 64)
    dummy_context = torch.randn(1, 16, 64)
    out = model(dummy_input, context=dummy_context)
    assert out.shape == (1, 256, 64)
