import dataclasses
import os
import pickle
import string
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.checkpointing.checkpoint_saving import CheckpointSaving
from modalities.checkpointing.stateful.app_state import AppState
from modalities.config.config import load_app_config_dict
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.create_packed_data import PackedDataGenerator
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.evaluator import Evaluator
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import NNModel
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer
from modalities.trainer import Trainer
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF

_ROOT_DIR = Path(__file__).parents[1]


@pytest.fixture
def dummy_packed_data_path(tmpdir) -> Path:
    data = b""
    header_size_in_bytes = 8
    token_size_in_bytes = 4
    tokens = list(range(20))
    data += (len(tokens) * token_size_in_bytes).to_bytes(header_size_in_bytes, byteorder="little")
    data += token_size_in_bytes.to_bytes(4, byteorder="little")
    data += b"".join([t.to_bytes(token_size_in_bytes, byteorder="little") for t in tokens])
    # NOTE: so far none of the implemented pytests use this index though!
    index = [(0, 24), (24, 40), (64, 12), (76, 4)]  # [(index,len), ...] -> in 4 bytes #lengths: 6,10,3,1
    data += pickle.dumps(index)
    dummy_packed_data_path = Path(tmpdir, "dummy.pbin")
    dummy_packed_data_path.write_bytes(data)
    return dummy_packed_data_path


@pytest.fixture
def dummy_config_path() -> Path:
    return _ROOT_DIR / Path("tests/test_yaml_configs/config_lorem_ipsum_fsdp1.yaml")


@pytest.fixture
def dummy_config(monkeypatch, dummy_config_path) -> dict:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    config_dict = load_app_config_dict(dummy_config_path, experiment_id="0")
    return config_dict


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
def dummy_data_path_long(tmpdir) -> DataPathCollection:
    source_raw_dummy_data_path = _ROOT_DIR / Path("./data/lorem_ipsum_long.jsonl")
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
def indexed_dummy_data_path_long(dummy_data_path_long) -> DataPathCollection:
    index_generator = IndexGenerator(dummy_data_path_long.raw_data_path)
    index_generator.create_index(dummy_data_path_long.index_path)
    return dummy_data_path_long


@pytest.fixture
def wrapped_gpt2_tokenizer() -> PreTrainedHFTokenizer:
    gpt2_tokenizer_folder_path = Path(__file__).parents[1] / Path("data", "tokenizer", "hf_gpt2")
    tokenizer = PreTrainedHFTokenizer(
        pretrained_model_name_or_path=gpt2_tokenizer_folder_path, max_length=None, truncation=None, padding=False
    )
    return tokenizer


@pytest.fixture(scope="function")
def checkpoint_saving_mock():
    return MagicMock(spec=CheckpointSaving)


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
def optimizer_with_param_groups_mock():
    mock_optimizer = MagicMock(spec=Optimizer, param_groups=[{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}])

    def custom_step_function(lr_decay_factor):
        # Iterate over each parameter group and update the lr based on some logic
        for param_group in mock_optimizer.param_groups:
            param_group["lr"] += -0.01
        return mock_optimizer

    mock_optimizer.step = custom_step_function
    # These are some hacks that fixes issues when the pytorch  LRScheduler super constructor
    # is implicitly instantiated. They seem to monkey patch the step function, making sure that
    # the lr scheduler step function is called after the optimizer step function.
    # See: https://github.com/pytorch/pytorch/blob/0b68a28c87df2c6eb2cf530be4659b5a2f8a95b0/torch/optim/lr_scheduler.py#L54
    mock_optimizer.step.__self__ = mock_optimizer
    mock_optimizer.step.__func__ = custom_step_function

    return mock_optimizer


@pytest.fixture(scope="function")
def scheduler_mock():
    mocked_lr_scheduler = MagicMock(spec=LRScheduler)
    mocked_lr_scheduler.get_last_lr = lambda: [0.0]
    return mocked_lr_scheduler


@pytest.fixture(scope="function")
def app_state_mock():
    return MagicMock(spec=AppState)


@pytest.fixture(scope="function")
def gradient_clipper_mock():
    gradient_clipper = MagicMock(spec=GradientClipperIF)
    gradient_clipper.clip_gradients = lambda: torch.Tensor([0.0])
    return gradient_clipper


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
def trainer(progress_publisher_mock, gradient_clipper_mock):
    return Trainer(
        global_rank=int(os.getenv("RANK")),
        progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        gradient_acc_steps=1,
        gradient_clipper=gradient_clipper_mock,
        global_num_tokens_per_train_step=10,
        num_seen_train_steps=0,
        global_num_seen_tokens=0,
        num_target_tokens=100,
        num_target_steps=10,
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
    if torch.__version__ < "2.4":
        # see https://pytorch.org/docs/2.3/_modules/torch/cuda.html#device_count
        torch.cuda.device_count.cache_clear()
    else:
        # see https://pytorch.org/docs/2.4/_modules/torch/cuda.html#device_count
        torch.cuda._cached_device_count = None


@pytest.fixture
def encoding_set_up():
    # Define the vocabulary
    vocabulary = {char: idx for idx, char in enumerate(string.ascii_lowercase)}

    # Ensure num_bytes_per_token is valid
    num_bytes_per_token = PackedDataGenerator._get_required_num_of_bytes_to_repr(len(vocabulary))
    assert num_bytes_per_token == 1  # This assertion will fail within the test framework if incorrect

    return vocabulary, num_bytes_per_token
