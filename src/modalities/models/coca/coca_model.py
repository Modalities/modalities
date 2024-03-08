import math
from functools import partial
from typing import Annotated, Dict, Tuple

import torch
from einops import repeat
from pydantic import BaseModel, Field
from torch import nn

from modalities.models.coca.attention_pooling import AttentionPooling
from modalities.models.coca.multi_modal_decoder import MultiModalDecoder
from modalities.models.coca.text_decoder import TextDecoder
from modalities.models.gpt2.gpt2_model import ActivationType, GPT2Config, WeightInitailizationConfig
from modalities.models.model import NNModel
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from modalities.nn.attention import AttentionConfig


class TextDecoderConfig(BaseModel):
    sample_key: str
    prediction_key: str
    block_size: Annotated[int, Field(ge=1)]
    vocab_size: Annotated[int, Field(ge=1)]
    n_layer: Annotated[int, Field(ge=1)]
    n_head: Annotated[int, Field(ge=1)]
    n_embd: Annotated[int, Field(ge=1)]
    ffn_hidden: Annotated[int, Field(ge=1)]
    dropout: Annotated[float, Field(ge=0.0)]
    bias: bool
    attention_config: AttentionConfig
    activation: ActivationType
    epsilon: Annotated[float, Field(ge=0.0)]


class CoCaConfig(BaseModel):
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
    n_shared_layer: Annotated[int, Field(ge=1)]
    weight_init: WeightInitailizationConfig


class CoCa(NNModel):
    """CoCa

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
        text_decoder_config: GPT2Config,
        n_shared_layer: int,
        weight_init: WeightInitailizationConfig,
    ) -> None:
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
            n_layer=text_decoder_config.n_layer - n_shared_layer,
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
        self.multimodal_decoder = MultiModalDecoder(
            sample_key=text_embd_prediction_key,
            prediction_key=text_decoder_config.prediction_key,
            block_size=text_decoder_config.block_size,
            n_layer=n_shared_layer,
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

        # init all weights
        self.apply(partial(self._init_weights, weight_init=weight_init))
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=weight_init.mean, std=weight_init.std / math.sqrt(2 * text_decoder_config.n_layer)
                )

    def _init_weights(self, module: nn.Module, weight_init: WeightInitailizationConfig):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=weight_init.mean, std=weight_init.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=weight_init.mean, std=weight_init.std)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vision_embd, vision_cls_token = self._forward_encode_vision(inputs)
        text_embd, text_cls_token = self._forward_encode_text(inputs)
        logits = self._forward_decode(text_embd, vision_embd)
        return {
            self.prediction_key: logits,
            self.vision_cls_prediction_key: vision_cls_token,
            self.text_cls_prediction_key: text_cls_token,
        }

    def _forward_encode_vision(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_embd = self.vision_encoder(inputs)[self.vision_embd_prediction_key]
        queries = repeat(self.vision_queries, "n d -> b n d", b=vision_embd.shape[0])
        vision_embd = self.attn_pool(queries, context=vision_embd)
        vision_embd, vision_cls_token = vision_embd[:, :-1, :], vision_embd[:, -1:, :]
        return vision_embd, vision_cls_token

    def _forward_encode_text(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embd = self.text_decoder(inputs)[self.text_embd_prediction_key]
        text_embd, text_cls_token = text_embd[:, :-1, :], text_embd[:, -1:, :]
        return text_embd, text_cls_token

    def _forward_decode(self, text_embd: torch.Tensor, vision_embd: torch.Tensor) -> torch.Tensor:
        decoder_inputs = {
            self.text_embd_prediction_key: text_embd,
            "context": vision_embd,
        }
        decoder_outputs = self.multimodal_decoder(decoder_inputs)
        logits = decoder_outputs[self.multimodal_decoder.prediction_key]
        return logits
