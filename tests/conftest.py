import dataclasses
import os
import pickle
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import Optimizer
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from transformers import GPT2TokenizerFast

from modalities.checkpointing.checkpointing import CheckpointingIF
from modalities.config.config import load_app_config_dict
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.dataloader.samplers import ResumableBatchSampler
from modalities.evaluator import Evaluator
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import NNModel
from modalities.trainer import Trainer

_ROOT_DIR = Path(__file__).parents[1]


@pytest.fixture
def dummy_packed_data_path(tmpdir) -> Path:
    data = b""
    header_size_in_bytes = 8
    token_size_in_bytes = 4
    tokens = list(range(20))
    data += (len(tokens) * token_size_in_bytes).to_bytes(header_size_in_bytes, byteorder="big")
    data += token_size_in_bytes.to_bytes(4, byteorder="big")
    data += b"".join([t.to_bytes(token_size_in_bytes, byteorder="big") for t in tokens])
    index = [(4, 24), (28, 40), (68, 12), (80, 4)]  # [(index,len), ...] -> in 4 bytes #lengths: 6,10,3,1
    data += pickle.dumps(index)
    dummy_packed_data_path = Path(tmpdir, "dummy.pbin")
    dummy_packed_data_path.write_bytes(data)
    return dummy_packed_data_path


@pytest.fixture
def dummy_config(monkeypatch) -> Dict:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dummy_config_path = _ROOT_DIR / Path("config_files/config_lorem_ipsum.yaml")
    config_dict = load_app_config_dict(dummy_config_path)
    return config_dict, dummy_config_path


@dataclasses.dataclass
class DataPathCollection:
    raw_data_path: Path
    index_path: Path


@pytest.fixture
def dummy_data_path(tmpdir) -> DataPathCollection:
    source_raw_dummy_data_path = _ROOT_DIR / Path("./data/lorem_ipsum.jsonl")
    dummy_data_path = Path(tmpdir, source_raw_dummy_data_path.name)
    dummy_data_path.write_text(source_raw_dummy_data_path.read_text())
    index_path = LargeFileLinesReader.default_index_path(dummy_data_path)
    index_path.unlink(missing_ok=True)
    return DataPathCollection(raw_data_path=dummy_data_path, index_path=index_path)


@pytest.fixture
def indexed_dummy_data_path(dummy_data_path) -> DataPathCollection:
    index_generator = IndexGenerator(dummy_data_path.raw_data_path)
    index_generator.create_index(dummy_data_path.index_path)
    return dummy_data_path


@pytest.fixture
def gpt2_tokenizer() -> GPT2TokenizerFast:
    default_gpt2_tokenizer_path = Path(__file__).parents[1] / Path("data", "tokenizer", "tokenizer_gpt2.json")
    assert default_gpt2_tokenizer_path.is_file()
    return GPT2TokenizerFast(tokenizer_file=str(default_gpt2_tokenizer_path))


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
        local_rank=int(os.getenv("LOCAL_RANK")),
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        gradient_acc_steps=1,
    )


@pytest.fixture(scope="function")
def set_env_cpu(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    # TODO: does not really ensure cpu-only usage. Alternative could be to patch `torch.cuda.is_available() = False`
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    def torch_distributed_cleanup():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    torch_distributed_cleanup()
    # gloo for CPU testing with reduce
    torch.distributed.init_process_group(backend="gloo")

    # use this to destroy this fixture's effect after it being used
    yield "torch.distributed.destroy_process_group"

    # TODO: discuss with Mehdi and Max and explain the side effects here.
    torch_distributed_cleanup()
    # setting CUDA_VISIBLE_DEVICES to "" creates a cache entry, which prevents resetting this CPU-only setup.
    # therefore after finish using this fixture, we need to clear this cache
    torch.cuda.device_count.cache_clear()


@pytest.fixture(scope="function")
def resumable_batch_sampler() -> ResumableBatchSampler:
    data_source = list(range(12))[::-1]  # torch.range(0,11)[::-1].reshape(3, 4)
    seq_sampler = SequentialSampler(data_source=data_source)

    seq_sampler = BatchSampler(sampler=seq_sampler, batch_size=3, drop_last=False)
    sampler = ResumableBatchSampler(start_index=2, underlying_batch_sampler=seq_sampler)
    return sampler
