import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from modalities.models.components.layer_norms import RMSLayerNorm, ZLayerNorm


@pytest.fixture
def rms_layer_norm() -> RMSLayerNorm:
    norm = RMSLayerNorm(ndim=3, epsilon=1e-6)
    weight_tensor = torch.Tensor([1, 2, 3])
    norm.weight = nn.Parameter(weight_tensor)
    return norm


@pytest.fixture
def z_layer_norm() -> ZLayerNorm:
    norm = ZLayerNorm(ndim=3, bias=True, epsilon=1e-6)
    return norm


def test_rms_layer_norm_forward(rms_layer_norm):
    x = torch.Tensor([0.1, 0.2, 0.3])
    output = rms_layer_norm(x)
    ref_x = x / np.sqrt((0.1**2 + 0.2**2 + 0.3**2) / 3 + 1e-6)
    ref_tensor = ref_x * rms_layer_norm.weight

    assert output.shape == x.shape
    assert all(output == ref_tensor)


def test_z_layer_norm_forward(z_layer_norm):
    x = torch.Tensor([0.1, 0.2, 0.3])
    output = z_layer_norm(x)
    ndim = z_layer_norm.weight.shape[0]
    ref_tensor = F.layer_norm(
        input=x,
        normalized_shape=z_layer_norm.weight.shape,
        weight=nn.Parameter(torch.ones(ndim)),
        bias=nn.Parameter(torch.zeros(ndim)) if z_layer_norm.bias_tensor is not None else None,
        eps=z_layer_norm.epsilon,
    )

    assert output.shape == x.shape
    assert all(output == ref_tensor)
