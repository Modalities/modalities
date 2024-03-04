import math
from functools import partial
from typing import Annotated

import torch
from einops import repeat
from pydantic import BaseModel, Field
from torch import nn

from modalities.models.coca.multi_modal_decoder import MultiModalDecoder
from modalities.models.coca.text_decoder import TextDecoder
from modalities.models.gpt2.gpt2_model import GPT2Config, WeightInitailizationConfig
from modalities.models.model import NNModel
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from modalities.nn.attention_pooling import AttentionPooling


class CoCaConfig(BaseModel):
    prediction_key: str = "logits"
    vision_embd_prediciton_key: str  # same key as vision encoder
    text_embd_prediciton_key: str  # same key as text encoder
    vision_cls_prediciton_key: str
    text_cls_prediciton_key: str
    vision_encoder_config: VisionTransformerConfig
    text_decoder_config: GPT2Config
    n_pool_head: Annotated[int, Field(ge=1)]
    n_vision_queries: Annotated[int, Field(ge=1)]
    bias_attn_pool: bool
    epsilon_attn_pool: Annotated[float, Field(ge=0.0)]
    n_shared_layer: Annotated[int, Field(ge=1)]


class CoCa(NNModel):
    """Contrastive Captioner"""

    def __init__(
        self,
        prediction_key: str,
        vision_cls_prediciton_key: str,
        text_cls_prediciton_key: str,
        vision_embd_prediciton_key: str,
        text_embd_prediciton_key: str,
        n_vision_queries: int,
        n_pool_head: int,
        bias_attn_pool: bool,
        epsilon_attn_pool: float,
        vision_encoder_config: VisionTransformerConfig,
        text_decoder_config: GPT2Config,
        n_shared_layer: int,
    ) -> None:
        super().__init__()
        self.prediction_key = prediction_key
        self.vision_cls_prediciton_key = vision_cls_prediciton_key
        self.text_cls_prediciton_key = text_cls_prediciton_key
        self.vision_embd_prediciton_key = vision_embd_prediciton_key
        self.text_embd_prediciton_key = text_embd_prediciton_key
        self.vision_encoder = VisionTransformer(**dict(vision_encoder_config))

        shared_decoder_kwargs = dict(text_decoder_config)
        del shared_decoder_kwargs["sample_key"]
        del shared_decoder_kwargs["prediction_key"]
        del shared_decoder_kwargs["block_size"]
        del shared_decoder_kwargs["n_layer"]
        del shared_decoder_kwargs["weight_init"]

        self.text_decoder = TextDecoder(
            sample_key=text_decoder_config.sample_key,
            prediction_key=text_embd_prediciton_key,
            block_size=text_decoder_config.block_size + 1,  # +1 for the class token
            n_layer=text_decoder_config.n_layer - n_shared_layer,
            **shared_decoder_kwargs,
        )
        self.multimodal_decoder = MultiModalDecoder(
            sample_key=text_embd_prediciton_key,
            prediction_key=text_decoder_config.prediction_key,
            block_size=text_decoder_config.block_size,
            n_layer=n_shared_layer,
            **shared_decoder_kwargs,
        )

        # TODO Validate if weight tying is useful for coca
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
        )

        # init all weights
        weight_init = text_decoder_config.weight_init
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

    def forward(self, inputs):
        vision_embd, vision_cls_token = self._forward_encode_vision(inputs)
        text_embd, text_cls_token = self._forward_encode_text(inputs)
        logits = self._forward_decode(text_embd, vision_embd)
        return {
            self.prediction_key: logits,
            self.vision_cls_prediciton_key: vision_cls_token,
            self.text_cls_prediciton_key: text_cls_token,
        }

    def _forward_encode_vision(self, inputs):
        vision_embd = self.vision_encoder(inputs)[self.vision_embd_prediciton_key]
        queries = repeat(self.vision_queries, "n d -> b n d", b=vision_embd.shape[0])
        vision_embd = self.attn_pool(queries, context=vision_embd)
        vision_cls_token, vision_embd = vision_embd[:, :1, :], vision_embd[:, 1:, :]
        return vision_embd, vision_cls_token

    def _forward_encode_text(self, inputs):
        text_embd = self.text_decoder(inputs)[self.text_embd_prediciton_key]
        text_embd, text_cls_token = text_embd[:, :-1, :], text_embd[:, -1:, :]
        return text_embd, text_cls_token

    def _forward_decode(self, text_embd, vision_embd):
        decoder_inputs = {
            self.text_embd_prediciton_key: text_embd,
            "context": vision_embd,
        }
        decoder_outputs = self.multimodal_decoder(decoder_inputs)
        logits = decoder_outputs[self.multimodal_decoder.prediction_key]
        return logits
