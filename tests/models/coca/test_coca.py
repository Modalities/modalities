from pathlib import Path

import pytest
import torch

from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.models.coca.coca_model import CoCa, CoCaConfig
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR

# shared config
N_EMBD = 768

# text_decoder_config
TEXT_DECODER_VOCAB_SIZE = 50_304
TEXT_DECODER_BLOCK_SIZE = 1_024

# vision_transformer_config
N_IMAGE_CLASSES = 1_000
IMG_SIZE = 224
N_IMG_CHANNELS = 3

# audio_transformer_config
AUDIO_BLOCK_SIZE = 500
N_MELS = 128
SUB_SAMPLING_FACTOR = 4


def dummy_image_sample():
    input_image = torch.randn(1, N_IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (1, TEXT_DECODER_BLOCK_SIZE))
    VISION = torch.tensor([1])
    return dict(
        images=input_image,
        input_ids=input_text,
        modality=VISION,
    )


def dummy_audio_sample():
    audio_features = torch.randn(1, AUDIO_BLOCK_SIZE * SUB_SAMPLING_FACTOR, N_MELS)
    audio_len = torch.tensor([N_IMAGE_CLASSES / SUB_SAMPLING_FACTOR]).type(torch.int16)
    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (1, TEXT_DECODER_BLOCK_SIZE))
    AUDIO = torch.tensor([0])
    return dict(
        audio=audio_features,
        feats_len=audio_len,
        input_ids=input_text,
        modality=AUDIO,
    )


@pytest.mark.parametrize(
    "yaml,dummy_sample",
    [
        ("tests/models/coca/coca_config_vision.yaml", dummy_image_sample()),
        ("tests/models/coca/coca_config_audio.yaml", dummy_audio_sample()),
    ],
)
def test_coca(yaml, dummy_sample):
    # Create model
    config_file_path = _ROOT_DIR / Path(yaml)
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    coca_config = CoCaConfig.model_validate(config_dict)
    model = CoCa(**dict(coca_config))

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Run one training step
    optimizer.zero_grad()
    out = model(dummy_sample)
    loss = out["logits"].sum()
    loss.backward()
    optimizer.step()

    # Test outputs
    assert "logits" in out
    assert "modality_cls" in out
    assert "text_cls" in out
    assert out["logits"].shape == (1, TEXT_DECODER_BLOCK_SIZE, TEXT_DECODER_VOCAB_SIZE)
    assert out["modality_cls"].shape == (1, N_EMBD)
    assert out["text_cls"].shape == (1, N_EMBD)


def test_coca_audio_vision_together():
    # Create model
    config_file_path = _ROOT_DIR / Path("coca/coca_config_av.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    coca_config = CoCaConfig.model_validate(config_dict)
    model = CoCa(**dict(coca_config))

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    audio_sample = dummy_audio_sample()
    image_sample = dummy_image_sample()

    for dummy_samples in [audio_sample, image_sample]:
        optimizer.zero_grad()
        out = model(dummy_samples)
        loss = out["logits"].sum()
        loss.backward()
        optimizer.step()

        assert "logits" in out
        assert "modality_cls" in out
        assert "text_cls" in out
        assert out["logits"].shape == (1, TEXT_DECODER_BLOCK_SIZE, TEXT_DECODER_VOCAB_SIZE)
        assert out["modality_cls"].shape == (1, N_EMBD)
        assert out["text_cls"].shape == (1, N_EMBD)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_e2e_coca_training_run_without_checkpoint(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    # Load config
    dummy_config_path = _ROOT_DIR / Path("config_files/training/config_example_coca.yaml")
    config_dict = load_app_config_dict(dummy_config_path)

    # Disable checkpointing
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_strategy"]["config"]["k"] = 0

    main = Main(config_dict, dummy_config_path)
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)
