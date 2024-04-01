import torch

from modalities.nn.mlp import MLP


def test_mlp_forward():
    model = MLP(in_features=64, hidden_features=256)
    dummy_input = torch.randn(1, 10, 64)
    out = model(dummy_input)
    assert out.shape == (1, 10, 64)
