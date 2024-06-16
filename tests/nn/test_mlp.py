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
    bias = True
    mlp = SwiGLU(n_embd, bias)

    hidden_dim = 1536
    assert mlp._get_hidden_dim(n_embd) == hidden_dim
    
    n_embd = 511
    assert mlp._get_hidden_dim(n_embd) == hidden_dim
    
    n_embd = 512

    # batch size x sequence length x embedding dim
    input_tensor = torch.randn(1, 1, n_embd)
    output_tensor = mlp(input_tensor)
    assert output_tensor.shape == (1, 1, n_embd)


    c_fc = nn.Linear(in_features=n_embd, out_features=hidden_dim, bias=bias)
    c_proj = nn.Linear(in_features=n_embd,out_features=hidden_dim,bias=bias)
    out_proj = nn.Linear(in_features=hidden_dim, out_features=n_embd,bias=bias)
    silu = nn.SiLU()
    mlp.c_fc = c_fc
    mlp.c_proj = c_proj
    mlp.out_proj = out_proj

    output_tensor = mlp(input_tensor)
    assert torch.all(output_tensor == out_proj(silu(c_fc(input_tensor)) * c_proj(input_tensor)))

