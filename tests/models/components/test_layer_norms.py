import numpy as np
import pytest
import torch
import torch.nn as nn

from modalities.models.components.layer_norms import RMSLayerNorm


@pytest.fixture
def rms_layer_norm() -> RMSLayerNorm:
    norm = RMSLayerNorm(ndim=3, epsilon=1e-6)
    weight_tensor = torch.Tensor([1, 2, 3])
    norm.weight = nn.Parameter(weight_tensor)
    norm.bias = nn.Parameter(torch.ones(3))
    return norm


def test_rms_layer_norm_forward(rms_layer_norm):
    x = torch.Tensor([0.1, 0.2, 0.3])
    output = rms_layer_norm(x)
    ref_x = x / np.sqrt((0.1**2 + 0.2**2 + 0.3**2) / 3 + 1e-6)
    ref_tensor = ref_x * rms_layer_norm.weight + torch.tensor([1, 1, 1])

    assert output.shape == x.shape
    assert all(output == ref_tensor)
