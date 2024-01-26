import math
from typing import Annotated, Dict

import torch
import xformers.ops as xops
from einops import repeat
from pydantic import BaseModel, Field
from torch import nn

from modalities.models.gpt2.gpt2_model import ActivationType, Block, GPT2Config, LayerNorm, TransformerMLP
from modalities.models.model import NNModel
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from modalities.nn.attention import Attention


class TextDecoder(NNModel):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.sample_key = config.sample_key
        self.prediction_key = config.prediction_key

        self.cls_token = nn.Parameter(torch.empty(1, 1, config.n_embd))
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
                wpe=nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            n_embd=config.n_embd,
                            bias=config.bias,
                            epsilon=config.epsilon,
                            activation=config.activation,
                            n_head=config.n_head,
                            attention=config.attention,
                            dropout=config.dropout,
                            block_size=config.block_size,
                            ffn_hidden=config.ffn_hidden,
                        )
                        for _ in range(config.n_layer)
                    ]
                ),
            )
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=config.weight_init.mean, std=config.weight_init.std / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=self.config.weight_init.mean, std=self.config.weight_init.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=self.config.weight_init.mean, std=self.config.weight_init.std)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = inputs[self.sample_key]
        device = input_ids.device
        b, t = input_ids.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t + 1, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(input_ids)
        tok_emb = torch.cat([tok_emb, self.cls_token.repeat(b, 1, 1)], dim=1)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        return {self.prediction_key: x}


class MultiModalBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        bias: bool,
        epsilon: float,
        activation: ActivationType,
        n_head: int,
        dropout: float,
        ffn_hidden: int,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)
        self.attn = Attention(n_embd, n_head, bias, is_causal=True, use_cross_attention=True)
        self.ln_2 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)

        if activation == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation == ActivationType.FUSED_SWIGLU:
            hidden_dim = 256 * ((int(2 * 4 * n_embd / 3) + 256 - 1) // 256)
            self.mlp = xops.SwiGLU(n_embd, hidden_dim, n_embd, bias=False)
        else:
            raise Exception("unimplemented activation")

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), context=context)
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiModalDecoder(NNModel):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.sample_key = config.sample_key
        self.prediction_key = config.prediction_key

        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList(
                    [
                        MultiModalBlock(
                            n_embd=config.n_embd,
                            bias=config.bias,
                            epsilon=config.epsilon,
                            activation=config.activation,
                            n_head=config.n_head,
                            dropout=config.dropout,
                            ffn_hidden=config.ffn_hidden,
                        )
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(ndim=config.n_embd, bias=config.bias, epsilon=config.epsilon),
            )
        )
        self.lm_head = nn.Linear(in_features=config.n_embd, out_features=config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=config.weight_init.mean, std=config.weight_init.std / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=self.config.weight_init.mean, std=self.config.weight_init.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=self.config.weight_init.mean, std=self.config.weight_init.std)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs[self.sample_key]
        for block in self.transformer.h:
            x = block(x, context=inputs["context"])
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return {self.prediction_key: logits}


class AttentionalPooling(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool,
        epsilon: float,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)
        self.attn = Attention(n_embd, n_head, use_cross_attention=True)
        self.ln_2 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)

    def forward(self, vision_embd: torch.Tensor, vision_queries: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(vision_embd)
        x = self.attn(vision_queries, context=x)
        x = self.ln_2(x)
        return x


class CoCaConfig(BaseModel):
    prediction_key: str = "logits"
    vision_embd_prediciton_key: str  # same key as vision encoder
    text_embd_prediciton_key: str  # same key as text encoder
    vision_cls_prediciton_key: str
    text_cls_prediciton_key: str
    vision_encoder_config: VisionTransformerConfig
    text_decoder_config: GPT2Config
    multimodal_decoder_config: GPT2Config
    n_pool_head: Annotated[int, Field(ge=1)]
    n_vision_queries: Annotated[int, Field(ge=1)]
    bias_attn_pool: bool
    epsilon_attn_pool: Annotated[float, Field(ge=0.0)]


class CoCa(NNModel):
    """Contrastive Captioner"""

    def __init__(self, config: CoCaConfig) -> None:
        super().__init__()
        self.prediction_key = config.prediction_key
        self.vision_cls_prediciton_key = config.vision_cls_prediciton_key
        self.text_cls_prediciton_key = config.text_cls_prediciton_key
        self.vision_embd_prediciton_key = config.vision_embd_prediciton_key
        self.text_embd_prediciton_key = config.text_embd_prediciton_key
        self.vision_encoder = VisionTransformer(config.vision_encoder_config)
        self.text_decoder = TextDecoder(config.text_decoder_config)
        self.multimodal_decoder = MultiModalDecoder(config.multimodal_decoder_config)

        # TODO Validate if weight tying is useful for coca
        self.text_decoder.transformer.wte.weight = (
            self.multimodal_decoder.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # vision_queries: 256 queries for multimodal cross attention and 1 as vision cls token for contrastive learning
        self.vision_queries = nn.Parameter(
            torch.randn(config.n_vision_queries + 1, config.vision_encoder_config.n_embd)
        )
        self.attn_pool = AttentionalPooling(
            n_embd=config.vision_encoder_config.n_embd,
            n_head=config.n_pool_head,
            bias=config.bias_attn_pool,
            epsilon=config.epsilon_attn_pool,
        )

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
        vision_embd = self.attn_pool(vision_embd, queries)
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
