
import pytest
from unittest.mock import MagicMock
import torch
from modalities.models.gpt2.huggingface_model import HuggingFaceModel
from modalities.config.config import HuggingFaceModelConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM


@pytest.fixture
def config():
    config = MagicMock(spec=HuggingFaceModelConfig)
    config.config = MagicMock()
    config.config.prediction_key = MagicMock()
    config.config.prediction_key.return_value = "logits"
    return config


@pytest.fixture(scope='session')
def tensor():
    return torch.tensor([[6, 8, 0, 7, 6, 6, 5, 6, 2, 0],
            [7, 6, 5, 8, 0, 5, 9, 5, 8, 8],
            [9, 5, 9, 9, 0, 3, 9, 1, 0, 4],
            [9, 1, 3, 2, 3, 9, 6, 2, 4, 1],
            [7, 9, 3, 9, 0, 3, 4, 0, 1, 4]])


@pytest.fixture
def model(tensor, config):
    model = MagicMock(spec=GPT2LLM)
    model.forward = MagicMock()
    model.forward.return_value = {config.config.prediction_key: tensor}
    return model


def test_forward(config, model, tensor):
    model = HuggingFaceModel(model=model, config=config)
    assert type(model.forward(tensor)) == torch.Tensor
