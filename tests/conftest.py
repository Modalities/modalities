from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import Optimizer

from llm_gym.checkpointing.checkpointing import CheckpointingIF
from llm_gym.dataset_loader import LLMDataLoader
from llm_gym.evaluator import Evaluator
from llm_gym.logging_broker.publisher import MessagePublisher
from llm_gym.loss_functions import Loss
from llm_gym.models.model import NNModel
from llm_gym.trainer import Trainer


@pytest.fixture
def checkpointing_mock():
    return MagicMock(spec=CheckpointingIF)


@pytest.fixture
def evaluator_mock():
    return MagicMock(spec=Evaluator)


@pytest.fixture
def nn_model_mock():
    return MagicMock(spec=NNModel)


@pytest.fixture
def optimizer_mock():
    return MagicMock(spec=Optimizer)


@pytest.fixture
def loss_mock():
    return MagicMock(spec=Loss, return_value=torch.rand(1, requires_grad=True))


@pytest.fixture
def llm_data_loader_mock():
    return MagicMock(spec=LLMDataLoader)


@pytest.fixture
def progress_publisher_mock():
    return MagicMock(spec=MessagePublisher)


@pytest.fixture
def trainer(progress_publisher_mock):
    return Trainer(
        local_rank=0,
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
    )


def set_env(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")
