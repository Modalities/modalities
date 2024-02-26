import torch

from modalities.nn.attention_pooling import AttentionalPooling


def test_attn_pool():
    model = AttentionalPooling(n_embd=768, n_head=8, bias=False, epsilon=1e-5)
    dummy_vision_embed = torch.randn(1, 256, 768)
    dummy_queries = torch.randn(1, 257, 768)
    out = model(dummy_vision_embed, dummy_queries)
    assert out.shape == (1, 257, 768)
