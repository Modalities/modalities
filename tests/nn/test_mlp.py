import torch

from modalities.models.model import SwiGLU
from modalities.nn.mlp import MLP


def test_mlp_forward():
    model = MLP(in_features=64, hidden_features=256)
    dummy_input = torch.randn(1, 10, 64)
    out = model(dummy_input)
    assert out.shape == (1, 10, 64)

def test_SwiGLU_forward():
    n_embd = 512
    bias = True
    model = SwiGLU(n_embd, bias)
    input_tensor = torch.randn(1, n_embd)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (1, n_embd)

