from unittest.mock import MagicMock

import pytest
import torch.distributed
import torch.nn as nn

import modalities.models.model_factory as mf
from modalities.models.model_factory import ModelFactory


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DummyBlock()
        self.block2 = DummyBlock()
        self.head = nn.Linear(10, 2)

    def forward(self, x):
        return self.head(self.block2(self.block1(x)))


@pytest.fixture
def dummy_model():
    return DummyModel()


def test_get_fsdp2_wrapped_model(dummy_model):
    def mock_get_module_class_from_name(model, name):
        assert name == "DummyBlock"
        return DummyBlock

    mock_fully_shard = MagicMock()

    mf.get_module_class_from_name = mock_get_module_class_from_name
    mf.fully_shard = mock_fully_shard

    torch.distributed.get_rank = lambda: 0

    mock_settings = MagicMock()
    mock_settings.param_dtype.value = "float16"
    mock_settings.reduce_dtype.value = "float16"

    mock_device_mesh = {"dp_shard": "fake-mesh"}
    mock_mesh = MagicMock()
    mock_mesh.__getitem__.side_effect = lambda x: mock_device_mesh[x]
    mock_mesh.get = mock_device_mesh.get

    result = ModelFactory.get_fsdp2_wrapped_model(
        model=dummy_model,
        block_names=["DummyBlock"],
        device_mesh=mock_mesh,
        mixed_precision_settings=mock_settings,
        reshard_after_forward=True,
    )

    assert mock_fully_shard.call_count == 3
    calls = mock_fully_shard.call_args_list
    assert calls[-1][1]["mesh"] == "fake-mesh"
    assert calls[-1][1]["reshard_after_forward"] is True
    # Because we are mocking fully_shard, we do not return a FSDP2 object
    assert isinstance(result, DummyModel)
