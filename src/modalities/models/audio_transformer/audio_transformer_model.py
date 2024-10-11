from typing import Annotated

import torch
from pydantic import BaseModel, Field
from torch import nn

from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention
from modalities.nn.mlp import MLP


class AudioTransformerConfig(BaseModel):
    """
    Configuration for an audio transformer model using conformer blocks.

    This configuration class defines all necessary parameters to instantiate and configure an `AudioTransformer` model.

    Args:
        sample_key (str): The key in the input dictionary that contains the audio samples.
        prediction_key (str): The key under which the model's output will be stored in the output dictionary.
        block_size (int): The size of each block for positional embeddings. Must be a positive integer.
        n_mels (int): The number of mel-frequency bands used for input audio feature extraction. 
            Must be a positive integer.
        n_embd (int): The embedding dimension used throughout the model. Must be a positive integer.
        n_heads (int): The number of attention heads in the conformer blocks. Must be a positive integer.
        n_conformer_blocks (int): The number of conformer blocks to include in the transformer model. 
            Must be a positive integer.
        attention_config (AttentionConfig): Configuration object for attention mechanisms.
        pointwise_conv_kernel_size (int): Kernel size for the pointwise convolutional layers in conformer blocks. 
            Must be a positive integer.
        depthwise_conv_kernel_size (int): Kernel size for the depthwise convolutional layers in conformer blocks. 
            Must be a positive integer.
        ffmodule_dropout (float, optional): Dropout rate for feed-forward modules in conformer blocks. 
            Must be a float less than 1.0. Default is 0.1.
        attn_dropout (float, optional): Dropout rate for attention mechanisms. Must be a float less than 1.0. 
            Default is 0.1.
        convmodule_dropout (float, optional): Dropout rate for depthwise convolutional layers in conformer blocks. 
            Must be a float less than 1.0. Default is 0.1.

    Returns:
        AudioTransformerConfig: A configuration object that can be used to instantiate an `AudioTransformer` model with\
            the specified parameters.

    Examples:
        >>> audio_encoder_config = AudioTransformerConfig(
            sample_key="audio",
            prediction_key="audio_embeddings",
            block_size=2_000,
            n_mels=128,
            n_embd=768,
            n_heads=8,
            n_conformer_blocks=2,
            attention_config=AttentionConfig(attention_engine_type="default_attention"),
            pointwise_conv_kernel_size=1,
            depthwise_conv_kernel_size=31
        )
    """

    sample_key: str
    prediction_key: str
    block_size: Annotated[int, Field(ge=1)]
    n_mels: Annotated[int, Field(ge=1)]
    n_embd: Annotated[int, Field(ge=1)]
    n_heads: Annotated[int, Field(ge=1)]
    n_conformer_blocks: Annotated[int, Field(ge=1)]
    attention_config: AttentionConfig
    pointwise_conv_kernel_size: Annotated[int, Field(ge=1)]
    depthwise_conv_kernel_size: Annotated[int, Field(ge=1)]
    ffmodule_dropout: Annotated[float, Field(lt=1.0)] = 0.1
    attn_dropout: Annotated[float, Field(lt=1.0)] = 0.1
    convmodule_dropout: Annotated[float, Field(lt=1.0)] = 0.1


class ConvolutionModule(nn.Module):
    """
    A convolutional module designed to process sequences using a series of layers including LayerNorm,
    pointwise convolutions, GLU activation, depthwise convolution, batch normalization, SiLU (Swish) activation,
    and a final pointwise convolution.
    """

    def __init__(
        self,
        n_embd: int,
        pointwise_conv_kernel_size: int,
        depthwise_conv_kernel_size: int,
        dropout: float,
    ):
        """
        Initializes the ConvolutionModule class.

        Args:
            n_embd (int): The number of embedding dimensions. Must be a positive integer.
            pointwise_conv_kernel_size (int): The kernel size for both the first and second pointwise convolutions.
            depthwise_conv_kernel_size (int): The kernel size for the depthwise convolution.
            dropout (float): Dropout rate applied after each layer. Must be a float between 0 and 1.

        Examples:
            >>> module = ConvolutionModule(
                n_embd=768,
                pointwise_conv_kernel_size=1,
                depthwise_conv_kernel_size=31,
                dropout=0.1
            )
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.pointwise_conv_1 = nn.Conv1d(
            n_embd,
            2 * n_embd,
            pointwise_conv_kernel_size,
            padding="same",
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            n_embd,
            n_embd,
            kernel_size=depthwise_conv_kernel_size,
            groups=n_embd,
            padding="same",
        )
        self.batch_norm = nn.BatchNorm1d(
            n_embd,
        )
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(
            n_embd,
            n_embd,
            pointwise_conv_kernel_size,
            padding="same",
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the convolutional module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D), where B is the batch size,
                T is the number of time steps, and D is the embedding dimension.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).
        """
        if x.shape[1] == 1:
            raise ValueError("The time dimension of the input to the convolution module cannot be 1!")

        x = self.ln_1(x)
        x = x.transpose(1, 2)
        x = self.glu(self.pointwise_conv_1(x))
        x = self.swish(self.batch_norm(self.depthwise_conv(x)))
        x = self.pointwise_conv_2(x)
        return self.dropout(x.transpose(1, 2))


class ConformerBlock(nn.Module):
    """
    This block combines self-attention, feed-forward modules, and depthwise convolutional layers to provide
    efficient processing of sequential data.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        attention_config: AttentionConfig,
        pointwise_conv_kernel_size: int,
        depthwise_conv_kernel_size: int,
        ffmodule_dropout: float,
        attn_dropout: float,
        convmodule_dropout: float,
    ) -> None:
        """Initializes the ConformerBlock class.

        Args:
            n_embd (int): The number of expected features in the input.
            n_heads (int): Number of parallel attention heads.
            attention_config (AttentionConfig): Configuration for the attention mechanism, typically a dictionary or \
                class instance.
            pointwise_conv_kernel_size (int): Kernel size of the depthwise convolutional layer.
            depthwise_conv_kernel_size (int): The kernel size for the depthwise convolutional module.
            ffmodule_dropout (float): Dropout rate for feed-forward modules.
            attn_dropout (float): Dropout rate for attention mechanism.
            convmodule_dropout (float): Dropout rate for the convolutional module.
        """
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.entry_ffmodule = MLP(
            in_features=n_embd,
            act_fn=nn.SiLU,
            dropout=ffmodule_dropout,
        )
        self.ln_mhsa = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(
            attention_config=attention_config,
            attention_type=AttentionType.NON_CAUSAL_SELF_ATTENTION,
            n_embd=n_embd,
            n_head=n_heads,
            dropout=attn_dropout,
        )
        self.convmodule = ConvolutionModule(
            n_embd,
            pointwise_conv_kernel_size,
            depthwise_conv_kernel_size,
            convmodule_dropout,
        )
        self.ln_2 = nn.LayerNorm(
            n_embd,
        )
        self.exit_ffmodule = MLP(
            in_features=n_embd,
            act_fn=nn.SiLU,
            dropout=ffmodule_dropout,
        )
        self.exit_ln = nn.LayerNorm(
            n_embd,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the conformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D), where B is the batch size,
                T is the number of time steps, and D is the embedding dimension.
            mask (torch.Tensor): Attention mask of shape (N, 1, L) or (N, L, L), where N is the batch size,
                L is the sequence length. If not provided, no attention mask will be used.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).
        """
        x = self.ln_1(x)
        x = x + 0.5 * self.entry_ffmodule(x)
        x = x + self.attn(self.ln_mhsa(x), mask=mask)
        x = x + self.convmodule(x)
        x = self.ln_2(x)
        x = x + 0.5 * self.exit_ffmodule(x)
        return self.exit_ln(x)


class AudioTransformer(nn.Module):
    """An audio transformer model using conformer blocks for processing audio data and generating predictions.

    This model includes convolutional layers, subsampling, positional embeddings,
    and multiple conformer blocks for feature extraction and processing."""

    def __init__(
        self,
        *,
        sample_key: str,
        prediction_key: str,
        block_size: int,
        n_mels: int,
        n_embd: int,
        n_heads: int,
        n_conformer_blocks: int,
        attention_config: AttentionConfig,
        pointwise_conv_kernel_size: int,
        depthwise_conv_kernel_size: int,
        ffmodule_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        convmodule_dropout: float = 0.1,
    ):
        """
        Initializes the AudioTransformer model.

        Args:
            sample_key (str): The key in the input dictionary that contains the audio samples.
            prediction_key (str): The key under which the model's output will be stored in the output dictionary.
            block_size (int): The size of each block for positional embeddings.
            n_mels (int): The number of mel-frequency bands used for input audio feature extraction.
            n_embd (int): The embedding dimension used throughout the model.
            n_heads (int): The number of attention heads in the conformer blocks.
            n_conformer_blocks (int): The number of conformer blocks to include in the transformer model.
            attention_config (AttentionConfig): Configuration object for attention mechanisms.
            pointwise_conv_kernel_size (int): Kernel size for the pointwise convolutional layers in conformer blocks.
            depthwise_conv_kernel_size (int): Kernel size for the depthwise convolutional layers in conformer blocks.
            ffmodule_dropout (float): Dropout rate for feed-forward modules in conformer blocks. Default is 0.1.
            attn_dropout (float): Dropout rate for attention mechanisms. Default is 0.1.
            convmodule_dropout (float): Dropout rate for depthwise convolutional layers in conformer blocks.
                Default is 0.1.

        Examples:
            >>> audio_encoder_config = {
                "sample_key": "audio",
                "prediction_key": "audio_embeddings",
                "block_size": 2000,
                "n_mels": 128,
                "n_embd": 768,
                "n_heads": 8,
                "n_conformer_blocks": 2,
                "attention_config": {
                    "attention_engine_type": "default_attention"
                },
                "pointwise_conv_kernel_size": 1,
                "depthwise_conv_kernel_size": 31
            }
        """
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.block_size = block_size

        self.project = nn.Conv1d(in_channels=n_mels, out_channels=n_embd, kernel_size=3, padding="same")
        self.subsampler = nn.Sequential(
            nn.Conv1d(
                in_channels=n_embd,
                out_channels=n_embd,
                kernel_size=2,
                stride=2,
            ),
            nn.Conv1d(
                in_channels=n_embd,
                out_channels=n_embd,
                kernel_size=2,
                stride=2,
            ),
        )
        self.post_subsampler_linear = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.Dropout(0.1),
        )

        self.positional_embeddings = nn.Embedding(self.block_size, n_embd)
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    n_embd,
                    n_heads,
                    attention_config,
                    pointwise_conv_kernel_size,
                    depthwise_conv_kernel_size,
                    ffmodule_dropout,
                    attn_dropout,
                    convmodule_dropout,
                )
                for _ in range(n_conformer_blocks)
            ]
        )

    def forward(
        self,
        inputs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the AudioTransformer model.

        Args:
            inputs (dict[str, tuple[torch.Tensor, torch.Tensor]]): A dictionary containing the input tensors. 
                It must include the key specified by `sample_key`.

        Returns:
            dict[str, tuple[torch.Tensor, torch.Tensor]]: A dictionary with a single key specified by `prediction_key`,\
                containing the model's output.
        """
        x = inputs[self.sample_key]  # x.shape: B, T, D
        attn_key_mask = self._get_attn_key_mask(inputs["audio_len"])
        # x.shape: B, T, D
        x = self.project(x.transpose(1, 2))  # x.shape: B, D, T
        x = self.subsampler(x)  # x.shape: B, D, T/4
        x = x.transpose(1, 2)
        x = self.post_subsampler_linear(x)
        x = x + self.positional_embeddings.weight

        for block in self.conformer_blocks:
            x = block(x, attn_key_mask)
        return {self.prediction_key: x}

    def _get_attn_key_mask(
        self,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Generates an attention key mask based on input sequence lengths.
        stack = []
        for length in lengths:
            ones = torch.ones(length, self.block_size)
            ones[1:, length:] = 0
            stack.append(ones)
        return (
            torch.nn.utils.rnn.pad_sequence(
                stack + [torch.zeros(self.block_size, self.block_size)],
                batch_first=True,
            )
            .transpose(1, 2)[:-1]
            .unsqueeze_(1)
        ).to(lengths.device)
