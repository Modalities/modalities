import pytest
import torch
import torch.nn as nn

from modalities.exceptions import ModelStateError
from modalities.models.model_factory import ModelFactory


class AllMetaDeviceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, device="meta")
        self.register_buffer("buffer", torch.empty(1, device="meta"))


class AllRealDeviceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.register_buffer("buffer", torch.empty(1))


class MixedDeviceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, device="meta")
        self.register_buffer("buffer", torch.empty(1))  # Not on meta device


def test_is_model_on_meta_device_true():
    model = AllMetaDeviceModel()
    assert ModelFactory._is_model_on_meta_device(model)


def test_is_model_on_meta_device_false():
    model = AllRealDeviceModel()
    assert not ModelFactory._is_model_on_meta_device(model)


def test_is_model_on_meta_device_mixed_raises():
    model = MixedDeviceModel()
    with pytest.raises(ModelStateError):
        ModelFactory._is_model_on_meta_device(model)
