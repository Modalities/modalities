import os
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


@pytest.fixture(scope="function")
def checkpointing_mock():
    return MagicMock(spec=CheckpointingIF)


@pytest.fixture(scope="function")
def evaluator_mock():
    return MagicMock(spec=Evaluator)


@pytest.fixture(scope="function")
def nn_model_mock():
    return MagicMock(spec=NNModel)


@pytest.fixture(scope="function")
def optimizer_mock():
    return MagicMock(spec=Optimizer)


@pytest.fixture(scope="function")
def loss_mock():
    return MagicMock(spec=Loss, return_value=torch.rand(1, requires_grad=True))


@pytest.fixture(scope="function")
def llm_data_loader_mock():
    return MagicMock(spec=LLMDataLoader)


@pytest.fixture(scope="function")
def progress_publisher_mock():
    return MagicMock(spec=MessagePublisher)


@pytest.fixture(scope="function")
def trainer(progress_publisher_mock):
    return Trainer(
        local_rank=os.getenv("LOCAL_RANK"),
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
    )


def set_env_cpu(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    # cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # gloo for CPU testing with reduce
    torch.distributed.init_process_group(backend="gloo")
