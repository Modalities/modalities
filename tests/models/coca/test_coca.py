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
N_FRAMES = 16

# audio_transformer_config
AUDIO_BLOCK_SIZE = 500
N_MELS = 128
SUB_SAMPLING_FACTOR = 4

BATCH_SIZE = 2


def dummy_image_sample():
    input_image = torch.randn(BATCH_SIZE, N_IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (BATCH_SIZE, TEXT_DECODER_BLOCK_SIZE))
    return dict(
        images=input_image,
        input_ids=input_text,
    )


def dummy_video_sample():
    input_video = torch.randn(BATCH_SIZE, N_FRAMES, N_IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (BATCH_SIZE, TEXT_DECODER_BLOCK_SIZE))
    return dict(
        video=input_video,
        input_ids=input_text,
    )


def dummy_audio_sample():
    audio_features = torch.randn(BATCH_SIZE, AUDIO_BLOCK_SIZE * SUB_SAMPLING_FACTOR, N_MELS)
    audio_len = torch.tensor([N_IMAGE_CLASSES / SUB_SAMPLING_FACTOR]).type(torch.int16)
    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (BATCH_SIZE, TEXT_DECODER_BLOCK_SIZE))
    return dict(
        audio=audio_features,
        audio_len=audio_len,
        input_ids=input_text,
    )


def dummy_img_aud_vid_sample():
    # separate image, audio, and video datasets
    input_image = torch.randn(BATCH_SIZE, N_IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    audio_features = torch.randn(BATCH_SIZE, AUDIO_BLOCK_SIZE * SUB_SAMPLING_FACTOR, N_MELS)
    audio_len = torch.tensor([N_IMAGE_CLASSES / SUB_SAMPLING_FACTOR]).type(torch.int16)
    input_video = torch.randn(BATCH_SIZE, N_FRAMES, N_IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (BATCH_SIZE * 3, TEXT_DECODER_BLOCK_SIZE))
    return dict(
        images=input_image,
        audio=audio_features,
        audio_len=audio_len,
        video=input_video,
        input_ids=input_text,
    )


def dummy_aud_vid_sample():
    # single video dataset which contains audio
    audio_features = torch.randn(BATCH_SIZE, AUDIO_BLOCK_SIZE * SUB_SAMPLING_FACTOR, N_MELS)
    audio_len = torch.tensor([N_IMAGE_CLASSES / SUB_SAMPLING_FACTOR]).type(torch.int16)
    input_video = torch.randn(BATCH_SIZE, N_FRAMES, N_IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    input_text = torch.randint(0, TEXT_DECODER_VOCAB_SIZE, (BATCH_SIZE, TEXT_DECODER_BLOCK_SIZE))
    return dict(
        audio=audio_features,
        audio_len=audio_len,
        video=input_video,
        input_ids=input_text,
    )


@pytest.mark.parametrize(
    "yaml,dummy_sample",
    [
        ("tests/models/coca/coca_config_image.yaml", dummy_image_sample()),
        ("tests/models/coca/coca_config_audio.yaml", dummy_audio_sample()),
        ("tests/models/coca/coca_config_video.yaml", dummy_video_sample()),
        ("tests/models/coca/coca_config_img_aud_vid.yaml", dummy_img_aud_vid_sample()),
        ("tests/models/coca/coca_config_aud_vid.yaml", dummy_aud_vid_sample()),
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
    text_output_batch_size = 0
    if coca_config.audio_encoder_config:
        assert "audio_cls" in out
        assert out["audio_cls"].shape == (BATCH_SIZE, N_EMBD)
        if coca_config.individual_datasets:
            assert out["audio_text_cls"].shape == (BATCH_SIZE, N_EMBD)
        if not coca_config.is_audio_video:
            text_output_batch_size += BATCH_SIZE
    if coca_config.image_encoder_config:
        assert "image_cls" in out
        assert out["image_cls"].shape == (BATCH_SIZE, N_EMBD)
        if coca_config.individual_datasets:
            assert out["image_text_cls"].shape == (BATCH_SIZE, N_EMBD)
        text_output_batch_size += BATCH_SIZE
    if coca_config.video_encoder_config:
        assert "video_cls" in out
        assert out["video_cls"].shape == (BATCH_SIZE, N_EMBD)
        if coca_config.individual_datasets:
            assert out["video_text_cls"].shape == (BATCH_SIZE, N_EMBD)
        text_output_batch_size += BATCH_SIZE
    if not coca_config.individual_datasets:
        assert out["text_cls"].shape == (BATCH_SIZE, N_EMBD)
    assert out["logits"].shape == (text_output_batch_size, TEXT_DECODER_BLOCK_SIZE, TEXT_DECODER_VOCAB_SIZE)
    assert "logit_scale" in out


@pytest.mark.skip(
    reason="The test itself is fine, but it is not working in the CI pipeline, as the infrastructure"
    "does not have enough GPU RAM."
)
# @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
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

    main = Main(dummy_config_path)
    main.config_dict = config_dict

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)
