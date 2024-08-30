from typing import Annotated, Dict, Tuple

import torch
from einops import repeat
from pydantic import BaseModel, Field
from torch import nn

from modalities.models.coca.attention_pooling import AttentionPooling
from modalities.models.coca.multi_modal_decoder import MultiModalTextDecoder
from modalities.models.coca.text_decoder import TextDecoder
from modalities.models.model import ActivationType, NNModel
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from modalities.nn.attention import AttentionConfig


class TextDecoderConfig(BaseModel):
    """
    Configuration class for the TextDecoder.


    Args:
        sample_key (str): The key for the samples.
        prediction_key (str): The key for the predictions.
        block_size (int): The block size. Must be greater than or equal to 1.
        vocab_size (int): The vocabulary size. Must be greater than or equal to 1.
        n_layer_text (int): The number of layers for processing text. Must be greater than or equal to 1.
        n_layer_multimodal_text (int): -. Must be greater than or equal to 1.
        n_head (int): The number of attention heads. Must be greater than or equal to 1.
        n_embd (int): The embedding size. Must be greater than or equal to 1.
        ffn_hidden (int): The hidden size for the feed-forward network. Must be greater than or equal to 1.
        dropout (float): The dropout rate. Must be greater than or equal to 0.0.
        bias (bool): Flag indicating whether to include bias in the model.
        attention_config (AttentionConfig): The attention configuration.
        activation (ActivationType): The activation type.
        epsilon (float): The epsilon value. Must be greater than or equal to 0.0.
    """

    sample_key: str
    prediction_key: str
    block_size: Annotated[int, Field(ge=1)]
    vocab_size: Annotated[int, Field(ge=1)]
    n_layer_text: Annotated[int, Field(ge=1)]
    n_layer_multimodal_text: Annotated[int, Field(ge=1)]
    n_head: Annotated[int, Field(ge=1)]
    n_embd: Annotated[int, Field(ge=1)]
    ffn_hidden: Annotated[int, Field(ge=1)]
    dropout: Annotated[float, Field(ge=0.0)]
    bias: bool
    attention_config: AttentionConfig
    activation: ActivationType
    epsilon: Annotated[float, Field(ge=0.0)]


class CoCaConfig(BaseModel):
    """
    Configuration class for CoCa model.

    Args:
        prediction_key (str): The key for the predictions.
        vision_embd_prediction_key (str): The key for the vision embeddings.
        text_embd_prediction_key (str): The key for the text embeddings.
        vision_cls_prediction_key (str): The key for the vision cls token.
        text_cls_prediction_key (str): The key for the text cls token.
        vision_encoder_config (VisionTransformerConfig): Configuration for the vision encoder.
        text_decoder_config (TextDecoderConfig): Configuration for the text decoder.
        n_pool_head (int): Number of attention heads for pooling.
        n_vision_queries (int): Number of vision queries.
        bias_attn_pool (bool): Flag indicating whether to use bias in attention pooling.
        epsilon_attn_pool (float): Epsilon value for attention pooling.

    """

    prediction_key: str = "logits"
    vision_embd_prediction_key: str  # same key as vision encoder
    text_embd_prediction_key: str
    vision_cls_prediction_key: str
    text_cls_prediction_key: str
    vision_encoder_config: VisionTransformerConfig
    text_decoder_config: TextDecoderConfig
    n_pool_head: Annotated[int, Field(ge=1)]
    n_vision_queries: Annotated[int, Field(ge=1)]
    bias_attn_pool: bool
    epsilon_attn_pool: Annotated[float, Field(ge=0.0)]


class CoCa(NNModel):
    """
    CoCa model

    The Contrastive Captioner (CoCa) is an encoder-decoder model that integrates the concepts of CLIP
    and generative models such as SimVLM by using contrastive and captioning losses for training.

    Paper: `CoCa: Contrastive Captioners are Image-Text Foundation Models`
    Link: https://arxiv.org/abs/2205.01917
    """

    def __init__(
        self,
        prediction_key: str,
        vision_cls_prediction_key: str,
        text_cls_prediction_key: str,
        vision_embd_prediction_key: str,
        text_embd_prediction_key: str,
        n_vision_queries: int,
        n_pool_head: int,
        bias_attn_pool: bool,
        epsilon_attn_pool: float,
        vision_encoder_config: VisionTransformerConfig,
        text_decoder_config: TextDecoderConfig,
    ) -> None:
        """
        Initializes the CocaModel object.

        Args:
            prediction_key (str): The key for the predictions.
            vision_cls_prediction_key (str): The key for the vision cls token.
            text_cls_prediction_key (str): The key for the text cls token.
            vision_embd_prediction_key (str): The key for the vision embeddings.
            text_embd_prediction_key (str): The key for the text embeddings.

            n_vision_queries (int): The number of vision queries.
            n_pool_head (int): The number of pool heads.
            bias_attn_pool (bool): Flag indicating whether to use bias in attention pooling.
            epsilon_attn_pool (float): The epsilon value for attention pooling.
            vision_encoder_config (VisionTransformerConfig): The configuration for the vision encoder.
            text_decoder_config (TextDecoderConfig): The configuration for the text decoder.

        Returns:
            None
        """
        super().__init__()
        self.prediction_key = prediction_key
        self.vision_cls_prediction_key = vision_cls_prediction_key
        self.text_cls_prediction_key = text_cls_prediction_key
        self.vision_embd_prediction_key = vision_embd_prediction_key
        self.text_embd_prediction_key = text_embd_prediction_key

        self.vision_encoder = VisionTransformer(**dict(vision_encoder_config))
        self.text_decoder = TextDecoder(
            sample_key=text_decoder_config.sample_key,
            prediction_key=text_embd_prediction_key,
            block_size=text_decoder_config.block_size + 1,  # +1 for the class token
            n_layer=text_decoder_config.n_layer_text,
            vocab_size=text_decoder_config.vocab_size,
            n_head=text_decoder_config.n_head,
            n_embd=text_decoder_config.n_embd,
            ffn_hidden=text_decoder_config.ffn_hidden,
            dropout=text_decoder_config.dropout,
            bias=text_decoder_config.bias,
            attention_config=text_decoder_config.attention_config,
            activation=text_decoder_config.activation,
            epsilon=text_decoder_config.epsilon,
        )
        self.multimodal_decoder = MultiModalTextDecoder(
            sample_key=text_embd_prediction_key,
            prediction_key=text_decoder_config.prediction_key,
            block_size=text_decoder_config.block_size,
            n_layer=text_decoder_config.n_layer_multimodal_text,
            vocab_size=text_decoder_config.vocab_size,
            n_head=text_decoder_config.n_head,
            n_embd=text_decoder_config.n_embd,
            ffn_hidden=text_decoder_config.ffn_hidden,
            dropout=text_decoder_config.dropout,
            bias=text_decoder_config.bias,
            attention_config=text_decoder_config.attention_config,
            activation=text_decoder_config.activation,
            epsilon=text_decoder_config.epsilon,
        )

        self.text_decoder.transformer.wte.weight = (
            self.multimodal_decoder.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # vision_queries: 256 queries for multimodal cross attention and 1 as vision cls token for contrastive learning
        self.vision_queries = nn.Parameter(torch.randn(n_vision_queries + 1, vision_encoder_config.n_embd))
        self.attn_pool = AttentionPooling(
            n_embd=vision_encoder_config.n_embd,
            n_head=n_pool_head,
            bias=bias_attn_pool,
            epsilon=epsilon_attn_pool,
            attention_config=text_decoder_config.attention_config,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the CoCa model.

        Args:
            inputs (dict[str, torch.Tensor]): Input dictionary containing the tensors.

        Returns:
            dict[str, torch.Tensor]: Output dictionary.
        """
        vision_embd, vision_cls_token = self._forward_encode_vision(inputs)
        text_embd, text_cls_token = self._forward_encode_text(inputs)
        logits = self._forward_decode(text_embd, vision_embd)
        return {
            self.prediction_key: logits,
            self.vision_cls_prediction_key: vision_cls_token,
            self.text_cls_prediction_key: text_cls_token,
        }

    def _forward_encode_vision(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input image using the vision encoder.

        Args:
            inputs (dict[str, torch.Tensor]): Dictionary containing vision inputs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing encoded vision embeddings and classification token.
        """
        vision_embd = self.vision_encoder(inputs)[self.vision_embd_prediction_key]
        queries = repeat(self.vision_queries, "n d -> b n d", b=vision_embd.shape[0])
        vision_embd = self.attn_pool(queries, context=vision_embd)
        vision_embd, vision_cls_token = vision_embd[:, :-1, :], vision_embd[:, -1:, :]
        return vision_embd, vision_cls_token

    def _forward_encode_text(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input text using the text decoder.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary containing input tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the encoded text tensor
            and the classification token tensor.
        """
        text_embd = self.text_decoder(inputs)[self.text_embd_prediction_key]
        text_embd, text_cls_token = text_embd[:, :-1, :], text_embd[:, -1:, :]
        return text_embd, text_cls_token

    def _forward_decode(self, text_embd: torch.Tensor, vision_embd: torch.Tensor) -> torch.Tensor:
        """
        Perform forward decoding using the given text and vision embeddings.

        Args:
            text_embd (torch.Tensor): The text embeddings.
            vision_embd (torch.Tensor): The vision embeddings.

        Returns:
            torch.Tensor: The logits obtained from the multimodal decoder.
        """
        decoder_inputs = {
            self.text_embd_prediction_key: text_embd,
            "context": vision_embd,
        }
        decoder_outputs = self.multimodal_decoder(decoder_inputs)
        logits = decoder_outputs[self.multimodal_decoder.prediction_key]
        return logits
