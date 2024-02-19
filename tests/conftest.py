import dataclasses
import json
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torchaudio
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from transformers import GPT2TokenizerFast

from modalities.__main__ import load_app_config_dict
from modalities.checkpointing.checkpointing import CheckpointingIF
from modalities.config.config import AppConfig
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


@dataclasses.dataclass
class DataPathCollection:
    raw_data_path: Path
    index_path: Path


@pytest.fixture
def dummy_packed_data_path(tmpdir) -> Path:
    data = b""
    data_header_size_in_bytes = 8
    codecs_header_size_in_bytes = 8
    int_size_in_bytes = 4
    # data and codecs
    tokens = list(range(20))
    codecs_bytes = pickle.dumps(["HfTokenizerCodec"])
    # headers
    data += (len(tokens) * int_size_in_bytes).to_bytes(data_header_size_in_bytes, byteorder="big")
    data += len(codecs_bytes).to_bytes(codecs_header_size_in_bytes, byteorder="big")
    # data and codecs
    data += b"".join([t.to_bytes(int_size_in_bytes, byteorder="big") for t in tokens])
    data += codecs_bytes
    # index
    index = [(16, 24), (40, 28), (68, 12), (80, 16)]  # [(index,len), ...] -> in 4 bytes #lengths: 6,10,3,1
    data += pickle.dumps(index)
    # write to file
    dummy_packed_data_path = Path(tmpdir, "dummy.pbin")
    dummy_packed_data_path.write_bytes(data)
    return dummy_packed_data_path


@pytest.fixture
def dummy_config(monkeypatch) -> AppConfig:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dummy_config_path = _ROOT_DIR / Path("config_files/config_lorem_ipsum.yaml")
    config_dict = load_app_config_dict(dummy_config_path)
    app_config = AppConfig.model_validate(config_dict)
    return app_config


@pytest.fixture
def dummy_data_path(tmpdir) -> DataPathCollection:
    source_raw_dummy_data_path = _ROOT_DIR / Path("./data/lorem_ipsum.jsonl")
    dummy_data_path = Path(tmpdir, source_raw_dummy_data_path.name)
    dummy_data_path.write_text(source_raw_dummy_data_path.read_text())
    index_path = LargeFileLinesReader.default_index_path(dummy_data_path)
    index_path.unlink(missing_ok=True)
    return DataPathCollection(raw_data_path=dummy_data_path, index_path=index_path)


@pytest.fixture
def indexed_multimodal_dummy_data_path(tmpdir) -> DataPathCollection:
    base_path = Path(tmpdir, "image_data")
    img_base_path = Path(base_path, "images")
    audio_base_path = Path(base_path, "audios")

    base_path.mkdir(parents=True, exist_ok=True)
    img_base_path.mkdir(parents=True, exist_ok=True)
    audio_base_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(base_path, "data.jsonl")
    index_path = Path(base_path, "data.idx")
    img_paths = [Path(img_base_path, "img_%i.png" % i) for i in range(15)]
    audio_paths = [Path(audio_base_path, "audio_%i.wav" % i) for i in range(15)]

    # create random images and save them into the temp directory
    for img_path in img_paths:
        im = np.random.rand(100, 100, 3) * 255
        im = Image.fromarray(im.astype("uint8")).convert("RGB")
        im.save(img_path, "PNG")

    # create random spectrograms and save them into the temp directory
    NUM_CHANNELS = 1
    SAMPLING_RATE = 16000
    AUDIO_DUR_SECS = 5

    for audio_path in audio_paths:
        audio = torch.randn(NUM_CHANNELS, SAMPLING_RATE * AUDIO_DUR_SECS)
        torchaudio.save(audio_path, audio, SAMPLING_RATE)

    # create the jsonl file
    with data_path.open("w+") as f:
        for img_path in img_paths:
            f.write(
                json.dumps(
                    {
                        "img_path": img_path.absolute().as_posix(),
                        "audio_path": audio_path.absolute().as_posix(),
                        "text": (
                            f"This item refers to the image stored at {str(img_path)} and "
                            f"the spectrogram stored at {str(audio_path)}"
                        ),
                    }
                )
                + "\n"
            )
    # create the index file to the jsonl file
    IndexGenerator(data_path).create_index(index_path)

    return DataPathCollection(raw_data_path=data_path, index_path=index_path)


@pytest.fixture
def indexed_dummy_data_path(dummy_data_path) -> DataPathCollection:
    index_generator = IndexGenerator(dummy_data_path.raw_data_path)
    index_generator.create_index(dummy_data_path.index_path)
    return dummy_data_path


@pytest.fixture
def gpt2_tokenizer() -> GPT2TokenizerFast:
    default_gpt2_tokenizer_path = Path(__file__).parents[1] / Path("data", "tokenizer", "tokenizer.json")
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
        gradient_acc_step=1,
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
