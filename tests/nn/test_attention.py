import torch

from modalities.nn.attention import Attention


def test_attention_forward():
    model = Attention(n_embd=64, n_head=8)
    dummy_input = torch.randn(1, 256, 64)
    out = model(dummy_input)
    assert out.shape == (1, 256, 64)


def test_attention_with_cross_attention_forward():
    model = Attention(n_embd=64, n_head=8, use_cross_attention=True)
    dummy_input = torch.randn(1, 256, 64)
    dummy_context = torch.randn(1, 16, 64)
    out = model(dummy_input, context=dummy_context)
    assert out.shape == (1, 256, 64)
