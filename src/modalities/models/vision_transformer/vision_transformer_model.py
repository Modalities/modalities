from typing import Annotated, Dict, Optional

import torch
from einops import rearrange
from pydantic import BaseModel, Field
from torch import nn

from modalities.nn.attention import Attention
from modalities.nn.mlp import MLP


class VisionTransformerConfig(BaseModel):
    sample_key: str
    prediction_key: str
    block_size: Annotated[int, Field(ge=1)] = 197
    n_classes: Optional[Annotated[int, Field(ge=1)]] = 1000
    n_layer: Annotated[int, Field(ge=1)] = 12
    n_head: Annotated[int, Field(ge=1)] = 8
    n_embd: Annotated[int, Field(ge=1)] = 768
    dropout: Annotated[float, Field(ge=0.0)] = 0.0
    patch_size: Annotated[int, Field(ge=1)] = 16
    patch_stride: Annotated[int, Field(ge=1)] = 16
    n_img_channels: Annotated[int, Field(ge=1)] = 3
    add_cls_token: bool = True
    bias: bool = True


class ImagePatchEmbedding(nn.Module):
    def __init__(
        self,
        n_img_channels: int = 3,
        n_embd: int = 768,
        patch_size: int = 16,
        patch_stride: int = 16,
        add_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_img_channels, n_embd, kernel_size=patch_size, stride=patch_stride)
        self.cls_token = None
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, *_ = x.shape
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.repeat(B, 1, 1), x], dim=1)
        return x


class Block(nn.Module):
    def __init__(self, n_embd: int = 768, n_head: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attention = Attention(n_embd, n_head)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.sample_key = config.sample_key
        self.prediction_key = config.prediction_key

        self.embd = ImagePatchEmbedding(
            config.n_img_channels, config.n_embd, config.patch_size, config.patch_stride, config.add_cls_token
        )
        self.pos_embd = nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block() for _ in range(config.n_layer)])

        self.head = None
        if config.n_classes is not None:
            self.norm = nn.LayerNorm(config.n_embd)
            self.head = nn.Linear(in_features=config.n_embd, out_features=config.n_classes, bias=False)

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embd(x)
        x = self.dropout(x + self.pos_embd.weight)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs[self.sample_key]
        x = self.forward_embeddings(x)
        if self.head:
            if self.embd.cls_token is not None:
                x = x[:, 0]
            else:
                x = x.mean(dim=1)
            x = self.head(self.norm(x))
        return {self.prediction_key: x}