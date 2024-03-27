import pytest
import torch

from modalities.models.audio_transformer.audio_transformer_model import AudioTransformer


@pytest.fixture
def pre_conformer_config():
    return {
        "input_dims": 80,
        "dropout": 0.1,
    }


@pytest.fixture
def audio_transformer_config():
    return {
        "sample_key": "audio_feats",
        "prediction_key": "audio_embeddings",
        "n_heads": 4,
        "n_embd": 512,
        "n_layers": 2,
        "depthwise_conv_kernel_size": 3,
        "dropout": 0.1,
    }


@pytest.fixture
def audio_transformer(
    pre_conformer_config,
    audio_transformer_config,
):
    return AudioTransformer(
        sample_key=audio_transformer_config["sample_key"],
        prediction_key=audio_transformer_config["prediction_key"],
        input_dims=pre_conformer_config["input_dims"],
        n_heads=audio_transformer_config["n_heads"],
        n_embd=audio_transformer_config["n_embd"],
        n_layers=audio_transformer_config["n_layers"],
        depthwise_conv_kernel_size=audio_transformer_config["depthwise_conv_kernel_size"],
        pre_conformer_dropout=pre_conformer_config["dropout"],
        conformer_dropout=audio_transformer_config["dropout"],
    )


@pytest.fixture
def dummy_input_div4():
    return {"audio_feats": (torch.randn(4, 80, 1000), torch.Tensor([1000 / 4] * 4))}


@pytest.fixture
def dummy_input_notdiv4():
    return {"audio_feats": (torch.randn(4, 80, 750), torch.Tensor([750 // 4] * 4))}


def test_audio_transformer_output_shape_div4(
    dummy_input_div4,
    audio_transformer,
    audio_transformer_config,
):
    output = audio_transformer(dummy_input_div4)
    audio_embeddings, audio_lengths = output[audio_transformer_config["prediction_key"]]
    assert audio_embeddings.shape[0] == 4
    assert audio_embeddings.shape[1] == 1000 / 4
    assert audio_embeddings.shape[2] == 512


def test_audio_transformer_output_shape_notdiv4(
    dummy_input_notdiv4,
    audio_transformer,
    audio_transformer_config,
):
    output = audio_transformer(dummy_input_notdiv4)
    audio_embeddings, audio_lengths = output[audio_transformer_config["prediction_key"]]
    assert audio_embeddings.shape[0] == 4
    assert audio_embeddings.shape[1] == 750 // 4
    assert audio_embeddings.shape[2] == 512
