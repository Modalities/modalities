import torch

from modalities.models.coca.attention_pooling import AttentionPooling


def test_attention_pooling_forward():
    model = AttentionPooling(n_embd=768, n_head=8, bias=False, epsilon=1e-5)
    dummy_input = torch.randn(1, 256, 768)
    dummy_queries = torch.randn(1, 257, 768)
    out = model(dummy_queries, dummy_input)
    assert out.shape == (1, 257, 768)
