import torch.nn as nn

from modalities.models.components.layer_norms import RMSLayerNorm
from modalities.models.components.norms import NormWrapperConfig


def test_norm_wrapper_builds_pytorch_rms_norm():
    config = NormWrapperConfig.model_validate(
        {"norm_type": "pytorch_rms_norm", "config": {"normalized_shape": 16, "eps": 1e-5}}
    )
    norm = config.build()
    assert isinstance(norm, nn.RMSNorm)
    assert norm.normalized_shape == (16,)
    assert norm.eps == 1e-5


def test_norm_wrapper_builds_layer_norm():
    config = NormWrapperConfig.model_validate(
        {"norm_type": "layer_norm", "config": {"normalized_shape": 8, "eps": 1e-6, "bias": False}}
    )
    norm = config.build()
    assert isinstance(norm, nn.LayerNorm)
    assert norm.bias is None


def test_norm_wrapper_builds_deprecated_rms_norm():
    config = NormWrapperConfig.model_validate({"norm_type": "rms_norm", "config": {"ndim": 8}})
    assert isinstance(config.build(), RMSLayerNorm)


def test_norm_wrapper_build_returns_a_fresh_instance_per_call():
    # Layers must not share normalization parameters, so build() has to create a new module
    # every time it is called.
    config = NormWrapperConfig.model_validate(
        {"norm_type": "pytorch_rms_norm", "config": {"normalized_shape": 4, "eps": 1e-5}}
    )
    first, second = config.build(), config.build()
    assert first is not second
    assert first.weight is not second.weight
