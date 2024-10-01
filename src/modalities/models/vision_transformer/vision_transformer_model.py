from math import floor
from typing import Annotated, Optional

import torch
from einops.layers.torch import Rearrange
from pydantic import BaseModel, Field
from torch import nn

from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention
from modalities.nn.mlp import MLP


class VisionTransformerConfig(BaseModel):
    """
    Configuration class for the VisionTransformer.


    Args:
        sample_key (str): The key for the input sample.
        prediction_key (str): The key for the model prediction.
        img_size (tuple[int, int] | int optional): The size of the input image. Defaults to 224.
        n_classes (int, optional): The number of output classes. Defaults to 1000.
        n_layer (int): The number of layers in the model. Defaults to 12.
        attention_config (AttentionConfig, optional): The configuration for the attention mechanism. Defaults to None.
        n_head (int): The number of attention heads. Defaults to 8.
        n_embd (int): The dimensionality of the embedding. Defaults to 768.
        ffn_hidden (int): The number of hidden units in the feed-forward network. Defaults to 3072.
        dropout (float): The dropout rate. Defaults to 0.0.
        patch_size (int): The size of the image patches. Defaults to 16.
        patch_stride (int): The stride of the image patches. Defaults to 16.
        n_img_channels (int): The number of image channels. Defaults to 3.
        add_cls_token (bool): Flag indicating whether to add a classification token. Defaults to True.
        bias (bool): Flag indicating whether to include bias terms. Defaults to True.
        num_video_frames (int): the number of video frames in case of video input
        n_latents: the number of latent queries used for the Perceiver block in case of video input. Defaults to 64.
    """

    sample_key: str
    prediction_key: str
    img_size: Annotated[tuple[int, int] | int, Field(ge=1)] = 224
    n_classes: Optional[Annotated[int, Field(ge=1)]] = 1000
    n_layer: Annotated[int, Field(ge=1)] = 12
    attention_config: AttentionConfig = None
    n_head: Annotated[int, Field(ge=1)] = 8
    n_embd: Annotated[int, Field(ge=1)] = 768
    ffn_hidden: Annotated[int, Field(ge=1)] = 3072
    dropout: Annotated[float, Field(ge=0.0)] = 0.0
    patch_size: Annotated[int, Field(ge=1)] = 16
    patch_stride: Annotated[int, Field(ge=1)] = 16
    n_img_channels: Annotated[int, Field(ge=1)] = 3
    add_cls_token: bool = True
    bias: bool = True
    num_video_frames: Annotated[int, Field(ge=0)] = 1
    n_latents: Annotated[int, Field(ge=1)] = 64


class ImagePatchEmbedding(nn.Module):
    """ImagePatchEmbedding class."""

    def __init__(
        self,
        n_img_channels: int = 3,
        n_embd: int = 768,
        patch_size: int = 16,
        patch_stride: int = 16,
        add_cls_token: bool = True,
    ) -> None:
        """
        Initializes an ImagePatchEmbedding object.


        Args:
            n_img_channels (int): Number of image channels. Defaults to 3.
            n_embd (int): Number of embedding dimensions. Defaults to 768.
            patch_size (int): Patch size for convolutional layer. Defaults to 16.
            patch_stride (int): Patch stride for convolutional layer. Defaults to 16.
            add_cls_token (bool): Flag indicating whether to add a classification token. Defaults to True.

        Returns:
            None
        """
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
        """
        Forward pass of the ImagePatchEmbedding.


        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        B = x.shape[0]
        x = self.conv(x)
        x = self.rearrange(x)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.repeat(B, 1, 1), x], dim=1)
        return x


class VideoPatchEmbedding(nn.Module):
    def __init__(
        self,
        n_img_channels: int = 3,
        n_embd: int = 768,
        patch_size: int = 16,
        patch_stride: int = 16,
    ) -> None:
        super().__init__()
        self.input_rearrange = Rearrange("b T c h w -> b c T h w")
        self.conv = nn.Conv3d(
            in_channels=n_img_channels,
            out_channels=n_embd,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_stride),
        )  # TODO: check the 3D conv again

        # See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
        self.rearrange = Rearrange("b c T h w -> b T (h w) c")  # TODO: this might change when implementing dataloader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_rearrange(x)
        x = self.conv(x)
        x = self.rearrange(x)
        return x  # [b T S D]


class VisionTransformerBlock(nn.Module):
    """VisionTransformerBlock class."""

    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 8,
        ffn_hidden: int = 3072,
        bias: bool = True,
        dropout: float = 0.0,
        attention_config: AttentionConfig = None,
    ) -> None:
        """
        Initializes a VisionTransformerBlock object.

        Args:
            n_embd (int, optional): The dimensionality of the embedding layer. Defaults to 768.
            n_head (int, optional): The number of attention heads. Defaults to 8.
            ffn_hidden (int, optional): The number of hidden units in the feed-forward network. Defaults to 3072.
            bias (bool, optional): Flag indicating whether to include bias terms. Defaults to True.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            attention_config (AttentionConfig, optional): The configuration for the attention mechanism.
            Defaults to None.

        Returns:
            None
        """
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
        """
        Forward pass of the VisionTransformerBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# TODO: extend to all modalities based on the original paper (https://arxiv.org/pdf/2103.03206)!
# TODO: extend this to work with video and images!
class PerceiverTransformerBlock(nn.Module):
    """Perceiver Resampler

    This is a transformer based architecture that performs cross and self attention to compress and embed video inputs.
    paper: 'Flamingo: a Visual Language Model for Few-Shot Learning'
    Link: https://github.com/mlfoundations/open_flamingo
    """

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
        self.norm_latents = nn.LayerNorm(n_embd)
        self.norm = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            attention_config=attention_config,
            attention_type=AttentionType.CROSS_ATTENTION,
        )
        self.mlp = MLP(in_features=n_embd, hidden_features=ffn_hidden, bias=bias, dropout=dropout)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        latents = self.norm_latents(latents)
        x = self.norm(x)
        context = torch.cat((x, latents), dim=-2)  # video features and the latent together
        latents = latents + self.attention(latents, context=context)
        latents = latents + self.mlp(latents)
        return latents


class VisionTransformer(nn.Module):
    """
    VisionTransformer class.

    The Vision Transformer (ViT) is a pure transformer architecture
    that applies attention mechanisms directly to sequences of image patches for image classification tasks.

    Paper: `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
    Link: https://arxiv.org/abs/2010.11929

    This architecture is extended to encode videos using a perceiver resampler transformer model
    """

    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        img_size: tuple[int, int] | int = 224,
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
        num_video_frames: int = 1,  # when dealing with video this is bigger than 1
        n_latents: int = 64,
    ) -> None:
        """
        Initializes the VisionTransformer object.

        Args:
            sample_key (str): The key for the samples.
            prediction_key (str): The key for the predictions.
            img_size (tuple[int, int] | int, optional): The size of the input image. Defaults to 224.
            n_classes (int, optional): The number of classes. Defaults to 1000.
            n_layer (int, optional): The number of layers. Defaults to 12.
            attention_config (AttentionConfig, optional): The attention configuration. Defaults to None.
            n_head (int, optional): The number of attention heads. Defaults to 8.
            n_embd (int, optional): The embedding dimension. Defaults to 768.
            ffn_hidden (int, optional): The hidden dimension of the feed-forward network. Defaults to 3072.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            patch_size (int, optional): The size of the image patch. Defaults to 16.
            patch_stride (int, optional): The stride of the image patch. Defaults to 16.
            n_img_channels (int, optional): The number of image channels. Defaults to 3.
            add_cls_token (bool, optional): Flag indicating whether to add a classification token. Defaults to True.
            bias (bool, optional): Flag indicating whether to include bias terms. Defaults to True.

            Returns:
                None
        """
        super().__init__()
        self.sample_key = sample_key
        self.has_cls_token = add_cls_token
        self.prediction_key = prediction_key
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.block_size = self._calculate_block_size(self.img_size, patch_size, patch_stride, add_cls_token)
        self.dropout = nn.Dropout(dropout)

        self.head = None
        if n_classes is not None:
            self.norm = nn.LayerNorm(n_embd)
            self.head = nn.Linear(in_features=n_embd, out_features=n_classes, bias=bias)

        self.vision_input = "Image"
        if num_video_frames > 1:  # video data
            self.vision_input = "Video"
            self.embedding_fn = VideoPatchEmbedding(n_img_channels, n_embd, patch_size, patch_stride)  # [b T S D]
            self.time_embd = nn.Parameter(torch.randn(num_video_frames, 1, n_embd))  # [T,1,d]
            if add_cls_token:
                n_latents += 1  # to count for a video level cls token
                self.block_size -= 1
            self.latents = nn.Parameter(torch.randn(n_latents, n_embd))  # [R,d]
            self.rearrange = Rearrange("b T S D -> b (T S) D")
        else:
            self.embedding_fn = ImagePatchEmbedding(n_img_channels, n_embd, patch_size, patch_stride, add_cls_token)

        self.positional_embedding_fn = nn.Embedding(
            num_embeddings=self.block_size, embedding_dim=n_embd
        )  # [S D] #TODO: this needs to be adjusted for video with cls_token
        block_classes = {"Video": PerceiverTransformerBlock, "Image": VisionTransformerBlock}

        self.blocks = nn.ModuleList(
            [
                block_classes[self.vision_input](
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

    def forward_images(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for processing images using the VisionTransformer module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after processing the input images.
        """
        x = self.embedding_fn(x)
        x = self.dropout(x + self.positional_embedding_fn.weight)
        for block in self.blocks:
            x = block(x)
        return x

    def forward_videos(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video data into a shorter sequence of tokens

        Args:
            x (torch.Tensor): images from multiple video frames
                shape (b c T h w)
                b: batch size
                T: temporal dim
                h,w: spatial dims (S=h*w)
                c: embedding dim (D)

        Returns:
            torch.Tensor: latents
                shape (b R D) R << T*S
        """
        x = self.embedding_fn(x)  # [b T S D]
        b, T = x.shape[:2]
        # TODO: check this!
        x = self.dropout(x + self.positional_embedding_fn.weight)
        x = self.dropout(x + self.time_embd.repeat(b, 1, 1, 1))
        x = self.rearrange(x)  # [b T*S D]
        latents = self.latents.repeat(b, 1, 1)  # [b,R,d] with R<<T*S
        for block in self.blocks:
            latents = block(x, latents)
        return latents

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # TODO video adapt
        """
        Forward pass of the VisionTransformer module.

        Args:
            inputs (dict[str, torch.Tensor]): Dictionary containing input tensors.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing output tensor.

        """
        x = inputs[self.sample_key]
        if self.vision_input == "Video":
            x = self.forward_videos(x)
        else:
            x = self.forward_images(x)
        if self.head:
            if self.has_cls_token:
                x = x[:, 0]
            else:
                x = x.mean(dim=1)
            x = self.head(self.norm(x))
        return {self.prediction_key: x}

    @staticmethod
    def _calculate_block_size(img_size: tuple[int, int], patch_size: int, patch_stride: int, add_cls_token: bool):
        """
        Calculates the block size.

        Args:
            img_size (tuple[int, int]): The size of the input image.
            patch_size (int): The size of each patch.
            patch_stride (int): The stride of each patch.
            add_cls_token (bool): Flag indicating whether to add a classification token.

        Returns:
            int: The calculated block size.
        """
        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for details
        block_size = (floor((img_size[0] - patch_size) / patch_stride) + 1) * (
            floor((img_size[1] - patch_size) / patch_stride) + 1
        ) + int(add_cls_token)
        return block_size
