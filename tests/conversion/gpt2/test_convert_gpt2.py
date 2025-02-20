import torch
import torch.nn as nn
from modalities.conversion.gpt2.convert_gpt2 import _copy_weights_base_modules


def test_copying_base_modules_weights_yields_identical_modules():
    m1 = nn.Linear(10, 10, bias=True)
    m2 = nn.Linear(10, 10, bias=True)
    m1.weight.data = torch.randn(10, 10)
    m1.bias.data = torch.randn(10)

    _copy_weights_base_modules(m1, m2)

    assert torch.equal(m1.weight.data, m2.weight.data)
    assert torch.equal(m1.bias.data, m2.bias.data)


def test_copying_base_modules_works_when_bias_is_false():
    m1 = nn.Linear(10, 10, bias=False)
    m2 = nn.Linear(10, 10, bias=False)
    m1.weight.data = torch.randn(10, 10)

    _copy_weights_base_modules(m1, m2)

    assert torch.equal(m1.weight.data, m2.weight.data)
    assert m1.bias == m2.bias == None
