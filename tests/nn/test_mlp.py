import torch
from torch import nn

from modalities.models.model import SwiGLU
from modalities.nn.mlp import MLP


def test_mlp_forward():
    model = MLP(in_features=64, hidden_features=256)
    dummy_input = torch.randn(1, 10, 64)
    out = model(dummy_input)
    assert out.shape == (1, 10, 64)


def test_SwiGLU_forward():
    n_embd = 512
    ffn_hidden = 4 * n_embd
    bias = True
    mlp = SwiGLU(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias)

    hidden_dim = 1536
    assert SwiGLU._get_hidden_dim(ffn_hidden=ffn_hidden) == hidden_dim

    n_embd = 511
    ffn_hidden = 4 * n_embd
    assert SwiGLU._get_hidden_dim(ffn_hidden=ffn_hidden) == hidden_dim

    n_embd = 512

    # batch size x sequence length x embedding dim
    input_tensor = torch.randn(1, 1, n_embd)
    output_tensor = mlp(input_tensor)
    assert output_tensor.shape == (1, 1, n_embd)

    W = nn.Linear(in_features=n_embd, out_features=hidden_dim, bias=bias)
    V = nn.Linear(in_features=n_embd, out_features=hidden_dim, bias=bias)
    W_2 = nn.Linear(in_features=hidden_dim, out_features=n_embd, bias=bias)
    silu = nn.SiLU()
    mlp.W = W
    mlp.V = V
    mlp.W_2 = W_2

    output_tensor = mlp(input_tensor)
    assert torch.all(output_tensor == W_2(silu(W(input_tensor)) * V(input_tensor)))
