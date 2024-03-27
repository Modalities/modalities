from pathlib import Path

import pytest
import torch

from modalities.__main__ import Main, load_app_config_dict
from modalities.models.coca.coca_model import CoCa, CoCaConfig
from tests.conftest import _ROOT_DIR


def dummy_image_sample():
    input_image = torch.randn(1, 3, 224, 224)
    text_decoder_vocab_size = 50304
    text_decoder_block_size = 1024
    input_text = torch.randint(0, text_decoder_vocab_size, (1, text_decoder_block_size))
    VISION = torch.tensor([1])
    return dict(
        images=input_image,
        input_ids=input_text,
        modality=VISION,
    )


def dummy_audio_sample():
    audio_features = torch.randn(1, 128, 1000)
    audio_len = torch.Tensor([1000 / 4])
    text_decoder_vocab_size = 50304
    text_decoder_block_size = 1024
    input_text = torch.randint(0, text_decoder_vocab_size, (1, text_decoder_block_size))
    AUDIO = torch.tensor([0])
    return dict(
        audio=(audio_features, audio_len),
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
    assert out["logits"].shape == (1, 1024, 50304)
    assert out["modality_cls"].shape == (1, 1, 768)
    assert out["text_cls"].shape == (1, 1, 768)


def test_coca_audio_vision_together():
    # Create model
    config_file_path = _ROOT_DIR / Path("tests/models/coca/coca_config_av.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    coca_config = CoCaConfig.model_validate(config_dict)
    model = CoCa(**dict(coca_config))

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    audio_sample = dummy_audio_sample()
    image_sample = dummy_image_sample()

    # Run for image
    optimizer.zero_grad()
    out = model(image_sample)
    loss = out["logits"].sum()
    loss.backward()
    optimizer.step()

    # Run for audio
    optimizer.zero_grad()
    out = model(audio_sample)
    loss = out["logits"].sum()
    loss.backward()
    optimizer.step()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_e2e_coca_training_run_without_checkpoint(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    # Load config
    dummy_config_path = _ROOT_DIR / Path("config_files/config_example_coca.yaml")
    config_dict = load_app_config_dict(dummy_config_path)

    # Disable checkpointing
    config_dict["checkpointing"]["config"]["checkpointing_strategy"]["config"]["k"] = 0

    main = Main(config_dict, dummy_config_path)
    main.run()
