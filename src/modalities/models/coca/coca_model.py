import math
from functools import partial
from typing import Annotated, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from pydantic import BaseModel, Field
from torch import nn

from modalities.models.audio_transformer.audio_transformer_model import AudioTransformer, AudioTransformerConfig
from modalities.models.coca.attention_pooling import AttentionPooling
from modalities.models.coca.multi_modal_decoder import MultiModalTextDecoder
from modalities.models.coca.text_decoder import TextDecoder
from modalities.models.gpt2.gpt2_model import ActivationType, WeightInitializationConfig
from modalities.models.model import NNModel
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from modalities.nn.attention import AttentionConfig


class AVConfig(BaseModel):
    audio_transformer_config: AudioTransformerConfig
    vision_transformer_config: VisionTransformerConfig


class TextDecoderConfig(BaseModel):
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
    prediction_key: str = "logits"
    modality_key: str = "modality"
    modality_embd_prediction_key: str
    text_embd_prediction_key: str
    modality_cls_prediction_key: str
    text_cls_prediction_key: str
    logit_scale_prediction_key: str
    modality_encoder_config: AudioTransformerConfig | VisionTransformerConfig | AVConfig
    text_decoder_config: TextDecoderConfig
    n_pool_head: Annotated[int, Field(ge=1)]
    n_vision_queries: Annotated[int, Field(ge=1)] | None
    n_audio_queries: Annotated[int, Field(ge=1)] | None
    bias_attn_pool: bool
    epsilon_attn_pool: Annotated[float, Field(ge=0.0)]
    weight_init: WeightInitializationConfig


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
        modality_key: str,
        modality_embd_prediction_key: str,
        text_embd_prediction_key: str,
        logit_scale_prediction_key: str,
        modality_cls_prediction_key: str,
        text_cls_prediction_key: str,
        n_vision_queries: int,
        n_audio_queries: int,
        n_pool_head: int,
        bias_attn_pool: bool,
        epsilon_attn_pool: float,
        modality_encoder_config: VisionTransformerConfig | AudioTransformerConfig | AVConfig,
        text_decoder_config: TextDecoderConfig,
        weight_init: WeightInitializationConfig,
    ) -> None:
        super().__init__()

        self.AUDIO = 0
        self.VISION = 1

        self.prediction_key = prediction_key
        self.modality_key = modality_key
        self.modality_embd_prediction_key = modality_embd_prediction_key
        self.text_embd_prediction_key = text_embd_prediction_key
        self.logit_scale_prediction_key = logit_scale_prediction_key

        self.modality_cls_prediction_key = modality_cls_prediction_key
        self.text_cls_prediction_key = text_cls_prediction_key

        self.n_pool_head = n_pool_head
        self.bias_attn_pool = bias_attn_pool
        self.epsilon_attn_pool = epsilon_attn_pool
        self.text_decoder_config = text_decoder_config

        if isinstance(modality_encoder_config, VisionTransformerConfig):
            self.vision_encoder, self.vision_queries, self.vision_attn_pool = self._init_modality(
                VisionTransformer,
                modality_encoder_config,
                n_vision_queries,
            )
        elif isinstance(modality_encoder_config, AudioTransformerConfig):
            self.audio_encoder, self.audio_queries, self.audio_attn_pool = self._init_modality(
                AudioTransformer,
                modality_encoder_config,
                n_audio_queries,
            )
        else:
            self.vision_encoder, self.vision_queries, self.vision_attn_pool = self._init_modality(
                VisionTransformer,
                modality_encoder_config.vision_transformer_config,
                n_vision_queries,
            )
            self.audio_encoder, self.audio_queries, self.audio_attn_pool = self._init_modality(
                AudioTransformer,
                modality_encoder_config.audio_transformer_config,
                n_audio_queries,
            )

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

        # Logit scale for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # init all weights
        self.apply(partial(self._init_weights, weight_init=weight_init))
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=weight_init.mean,
                    std=weight_init.std
                    / math.sqrt(2 * (text_decoder_config.n_layer_text + text_decoder_config.n_layer_multimodal_text)),
                )

    def _init_modality(self, encoder_class, encoder_config, n_queries):
        encoder = encoder_class(**dict(encoder_config))
        queries = nn.Parameter(torch.randn(n_queries + 1, encoder_config.n_embd))
        attn_pool = AttentionPooling(
            n_embd=encoder_config.n_embd,
            n_head=self.n_pool_head,
            bias=self.bias_attn_pool,
            epsilon=self.epsilon_attn_pool,
            attention_config=self.text_decoder_config.attention_config,
        )
        return encoder, queries, attn_pool

    def _init_weights(self, module: nn.Module, weight_init: WeightInitializationConfig):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=weight_init.mean, std=weight_init.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=weight_init.mean, std=weight_init.std)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO: The "modality_key" needs to be implemented.
        if inputs[self.modality_key][0] == self.AUDIO:
            modality_embd, modality_cls_token = self._forward_encode_audio(inputs)
        if inputs[self.modality_key][0] == self.VISION:
            modality_embd, modality_cls_token = self._forward_encode_vision(inputs)
        text_embd, text_cls_token = self._forward_encode_text(inputs)
        logits = self._forward_decode(text_embd, modality_embd)
        return {
            self.prediction_key: logits,
            self.modality_cls_prediction_key: modality_cls_token,
            self.text_cls_prediction_key: text_cls_token,
            self.logit_scale_prediction_key: self.logit_scale.exp(),
        }

    def _forward_encode_vision(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_embd = self.vision_encoder(inputs)[self.modality_embd_prediction_key]
        queries = repeat(self.vision_queries, "n d -> b n d", b=vision_embd.shape[0])
        vision_embd = self.vision_attn_pool(queries, context=vision_embd)
        vision_embd, vision_cls_token = vision_embd[:, :-1, :], F.normalize(vision_embd[:, -1, :], dim=-1)
        return vision_embd, vision_cls_token

    def _forward_encode_audio(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_embd, _ = self.audio_encoder(inputs)[self.modality_embd_prediction_key]
        queries = repeat(self.audio_queries, "n d -> b n d", b=audio_embd.shape[0])
        audio_embd = self.audio_attn_pool(queries, context=audio_embd)
        audio_embd, audio_cls_token = audio_embd[:, :-1, :], F.normalize(audio_embd[:, -1:, :], dim=-1)
        return audio_embd, audio_cls_token

    def _forward_encode_text(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embd = self.text_decoder(inputs)[self.text_embd_prediction_key]
        text_embd, text_cls_token = text_embd[:, :-1, :], F.normalize(text_embd[:, -1, :], dim=-1)
        return text_embd, text_cls_token

    def _forward_decode(self, text_embd: torch.Tensor, modality_embd: torch.Tensor) -> torch.Tensor:
        decoder_inputs = {
            self.text_embd_prediction_key: text_embd,
            "context": modality_embd,
        }
        decoder_outputs = self.multimodal_decoder(decoder_inputs)
        logits = decoder_outputs[self.multimodal_decoder.prediction_key]
        return logits
