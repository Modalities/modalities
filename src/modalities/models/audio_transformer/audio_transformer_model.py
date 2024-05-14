from typing import Annotated, Dict

import torch
from pydantic import BaseModel, Field
from torch import nn
from torchaudio.models import Conformer


class AudioTransformerConfig(BaseModel):
    sample_key: str
    prediction_key: str
    input_dims: Annotated[int, Field(ge=1)]
    pre_conformer_dropout: Annotated[float, Field(lt=1.0)]
    conformer_dropout: Annotated[float, Field(lt=1.0)]
    n_heads: Annotated[int, Field(ge=1)]
    n_embd: Annotated[int, Field(ge=1)]
    n_layers: Annotated[int, Field(ge=1)]
    depthwise_conv_kernel_size: Annotated[int, Field(ge=1)]


class PreConformer(nn.Module):
    def __init__(
        self,
        *,
        n_input_dims: int,
        dropout: float,
    ):
        super().__init__()
        self.subsampler = nn.Sequential(
            nn.Conv1d(
                in_channels=n_input_dims,
                out_channels=n_input_dims,
                kernel_size=2,
                stride=2,
            ),
            nn.Conv1d(
                in_channels=n_input_dims,
                out_channels=n_input_dims,
                kernel_size=2,
                stride=2,
            ),
        )
        self.linear = nn.Linear(n_input_dims, n_input_dims)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.transpose(1, 2)  # x.shape: batch_size, n_input_dims, n_input_frames

        x = self.subsampler(x)  # x.shape: batch_size, n_input_dims, ceil(n_input_frames / 4)
        x = x.transpose(1, 2)
        x = self.linear(x)  # x.shape: batch_size, ceil(n_input_frames / 4), n_input_dims
        x = self.dropout(x)
        return x


class AudioTransformer(nn.Module):
    def __init__(
        self,
        *,
        sample_key: str,
        prediction_key: str,
        input_dims: int,
        n_heads: int,
        n_embd: int,
        n_layers: int,
        depthwise_conv_kernel_size: int,
        pre_conformer_dropout: float,
        conformer_dropout: float,
    ):
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.pre_conformer = PreConformer(
            n_input_dims=input_dims,
            dropout=pre_conformer_dropout,
        )

        self.conformer = Conformer(
            input_dim=input_dims,
            num_heads=n_heads,
            ffn_dim=n_embd,
            num_layers=n_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=conformer_dropout,
        )

        self.post_conformer = nn.Sequential(
            nn.Linear(
                input_dims,
                n_embd,
            ),
            nn.LayerNorm(n_embd),
        )

    def forward(
        self,
        inputs: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        x = inputs[self.sample_key]  # x.shape: batch_size, n_input_dims, n_input_frames
        x_length = inputs["feats_len"]
        x = self.pre_conformer(x)  # x.shape: batch_size, ceil(n_input_frames / 4), n_input_dims
        x, x_length = self.conformer(x, x_length)  # x.shape: batch_size, ceil(n_input_frames / 4), n_input_dims
        x = self.post_conformer(x)
        return {self.prediction_key: (x, x_length)}
