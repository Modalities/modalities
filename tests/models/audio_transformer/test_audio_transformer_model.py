import pytest
import torch

from modalities.models.audio_transformer.audio_transformer_model import (
    AudioTransformer,
    ConformerBlock,
    ConvolutionModule,
)
from modalities.nn.attention import AttentionConfig


@pytest.fixture
def params() -> dict:
    return {
        "sample_key": "audio",
        "prediction_key": "audio_embeddings",
        "block_size": 5,
        "n_mels": 1,
        "n_conformer_blocks": 1,
        "n_embd": 1,
        "n_heads": 1,
        "attention_config": AttentionConfig(attention_engine_type="pytorch_flash_attention"),
        "pointwise_conv_kernel_size": 1,
        "depthwise_conv_kernel_size": 1,
        "dropout": 0.1,
    }


@pytest.fixture
def audio_transformer_model(params) -> AudioTransformer:
    return AudioTransformer(
        sample_key=params["sample_key"],
        prediction_key=params["prediction_key"],
        block_size=params["block_size"],
        n_mels=params["n_mels"],
        n_conformer_blocks=params["n_conformer_blocks"],
        n_embd=params["n_embd"],
        n_heads=params["n_heads"],
        attention_config=params["attention_config"],
        pointwise_conv_kernel_size=params["pointwise_conv_kernel_size"],
        depthwise_conv_kernel_size=params["depthwise_conv_kernel_size"],
        ffmodule_dropout=params["dropout"],
        attn_dropout=params["dropout"],
        convmodule_dropout=params["dropout"],
    )


@pytest.fixture
def invalid_forward_input() -> torch.Tensor:
    return torch.randn((1, 1, 256))


@pytest.fixture
def forward_input() -> dict[str, torch.Tensor]:
    return {"x": torch.randn((1, 2, 1)), "mask": torch.ones((1, 2))}


def test_convolution_module_forward_return_shape(
    params,
    forward_input,
):
    convolution = ConvolutionModule(
        params["n_embd"],
        params["pointwise_conv_kernel_size"],
        params["depthwise_conv_kernel_size"],
        params["dropout"],
    )

    out = convolution(forward_input["x"])

    assert out.shape == (1, 2, 1)


def test_convolution_module_forward_raise(
    params,
    invalid_forward_input,
):
    convolution = ConvolutionModule(
        params["n_embd"],
        params["pointwise_conv_kernel_size"],
        params["depthwise_conv_kernel_size"],
        params["dropout"],
    )

    with pytest.raises(ValueError, match="The time dimension of the input to the convolution module cannot be 1!"):
        convolution(invalid_forward_input)


def test_conformer_forward(params, forward_input):
    conformer = ConformerBlock(
        params["n_embd"],
        params["n_heads"],
        params["attention_config"],
        params["pointwise_conv_kernel_size"],
        params["depthwise_conv_kernel_size"],
        params["dropout"],
        params["dropout"],
        params["dropout"],
    )

    conformer(forward_input["x"], forward_input["mask"])


def test_audio_transformer__get_attn_key_mask(audio_transformer_model):
    lengths = torch.tensor([3])

    CORRECT_MASK = torch.Tensor(
        [
            [
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ]
            ]
        ]
    )

    CREATED_MASK = audio_transformer_model._get_attn_key_mask(lengths)
    assert torch.equal(CORRECT_MASK, CREATED_MASK)
