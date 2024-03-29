from math import floor
from typing import Annotated, Dict, Optional, Tuple, Union

import torch
from einops.layers.torch import Rearrange
from pydantic import BaseModel, Field
from torch import nn

from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention
from modalities.nn.mlp import MLP


class VisionTransformerConfig(BaseModel):
    sample_key: str
    prediction_key: str
    img_size: Annotated[Union[Tuple[int, int], int], Field(ge=1)] = 224
    n_classes: Optional[Annotated[int, Field(ge=1)]] = 1000
    n_layer: Annotated[int, Field(ge=1)] = 12
    attention_config: AttentionConfig = None
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
        self.conv = nn.Conv2d(
            in_channels=n_img_channels, out_channels=n_embd, kernel_size=patch_size, stride=patch_stride
        )

        # Define a rearrangement operation to reshape the tensor from
        # batched 4D format (batch_size, channels, height, width) to
        # batched 3D format (batch_size, height*width, channels).
        # This is required to support torch.compile.
        # See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
        self.rearrange = Rearrange("b c h w -> b (h w) c")

        self.cls_token = None
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.conv(x)
        x = self.rearrange(x)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.repeat(B, 1, 1), x], dim=1)
        return x


class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 8,
        ffn_hidden: int = 3072,
        bias: bool = True,
        dropout: float = 0.0,
        attention_config: AttentionConfig = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            attention_config=attention_config,
            attention_type=AttentionType.NON_CAUSAL_SELF_ATTENTION,
        )
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(in_features=n_embd, hidden_features=ffn_hidden, bias=bias, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT

    The Vision Transformer (ViT) is a pure transformer architecture
    that applies attention mechanisms directly to sequences of image patches for image classification tasks.

    Paper: `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
    Link: https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        img_size: Union[Tuple[int, int], int] = 224,
        n_classes: int = 1000,
        n_layer: int = 12,
        attention_config: AttentionConfig = None,
        n_head: int = 8,
        n_embd: int = 768,
        ffn_hidden: int = 3072,
        dropout: float = 0.0,
        patch_size: int = 16,
        patch_stride: int = 16,
        n_img_channels: int = 3,
        add_cls_token: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.block_size = self._calculate_block_size(self.img_size, patch_size, patch_stride, add_cls_token)

        self.embedding_fn = ImagePatchEmbedding(n_img_channels, n_embd, patch_size, patch_stride, add_cls_token)
        self.positional_embedding_fn = nn.Embedding(num_embeddings=self.block_size, embedding_dim=n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    ffn_hidden=ffn_hidden,
                    bias=bias,
                    dropout=dropout,
                    attention_config=attention_config,
                )
                for _ in range(n_layer)
            ]
        )

        self.head = None
        if n_classes is not None:
            self.norm = nn.LayerNorm(n_embd)
            self.head = nn.Linear(in_features=n_embd, out_features=n_classes, bias=bias)

    def forward_images(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_fn(x)
        x = self.dropout(x + self.positional_embedding_fn.weight)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs[self.sample_key]
        x = self.forward_images(x)
        if self.head:
            if self.embedding_fn.cls_token is not None:
                x = x[:, 0]
            else:
                x = x.mean(dim=1)
            x = self.head(self.norm(x))
        return {self.prediction_key: x}

    @staticmethod
    def _calculate_block_size(img_size: Tuple[int, int], patch_size: int, patch_stride: int, add_cls_token: bool):
        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for details
        block_size = (floor((img_size[0] - patch_size) / patch_stride) + 1) * (
            floor((img_size[1] - patch_size) / patch_stride) + 1
        ) + int(add_cls_token)
        return block_size
