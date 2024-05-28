from typing import Annotated, Dict

import torch
from pydantic import BaseModel, Field
from torch import nn

from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention
from modalities.nn.mlp import MLP


class AudioTransformerConfig(BaseModel):
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
    def __init__(
        self,
        n_embd: int,
        pointwise_conv_kernel_size: int,
        depthwise_conv_kernel_size: int,
        dropout: int,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
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
        self.bn = nn.BatchNorm1d(
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
        x = self.ln(x)
        x = x.transpose(1, 2)
        x = self.glu(self.pointwise_conv_1(x))
        x = self.swish(self.bn(self.depthwise_conv(x)))
        x = self.pointwise_conv_2(x)
        return self.dropout(x.transpose(1, 2))  # shape: B, T, D


class ConformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        attention_config,
        pointwise_conv_kernel_size: int,
        depthwise_conv_kernel_size: int,
        ffmodule_dropout: float,
        attn_dropout: float,
        convmodule_dropout: float,
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.entry_ffmodule = MLP(
            in_features=n_embd,
            act_fn=nn.SiLU,
            dropout=ffmodule_dropout,
        )
        self.mhsa_ln = nn.LayerNorm(n_embd)
        self.mhsa = MultiHeadAttention(
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
        self.ln2 = nn.LayerNorm(
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
        x = self.ln1(x)  # x.shape: B, T, D
        x = x + 0.5 * self.entry_ffmodule(x)
        x = x + self.mhsa(self.mhsa_ln(x), mask=mask)
        x = x + self.convmodule(x)
        x = self.ln2(x)
        x = x + 0.5 * self.exit_ffmodule(x)
        return self.exit_ln(x)


class AudioTransformer(nn.Module):
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
        inputs: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        x = inputs[self.sample_key]  # x.shape: B, T, D
        attn_key_mask = self._get_attn_key_mask(inputs["feats_len"])
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
    ):
        return (
            torch.nn.utils.rnn.pad_sequence(
                [torch.ones(length, self.block_size) for length in lengths]
                + [torch.ones(self.block_size, self.block_size)],
                batch_first=True,
            )
            .transpose(1, 2)[:-1]
            .unsqueeze_(1)
        ).to(lengths.device)
