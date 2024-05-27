from functools import partial
from typing import Dict, List

import torch
import xformers.ops as xops
from torch import nn

from modalities.models.gpt2.gpt2_model import ActivationType
from modalities.models.model import NNModel
from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention
from modalities.nn.mlp import MLP
from transformers import PreTrainedTokenizer


class TransformerBlock(nn.Module):
    def __init__(
            self,
            n_embd: int,
            bias: bool,
            epsilon: float,
            activation: ActivationType,
            n_head: int,
            dropout: float,
            ffn_hidden: int,
            with_context: bool,
            attention_type: AttentionType,
            attention_config: AttentionConfig = None,
            add_extra_mlp: bool = False,
    ):
        super().__init__()
        self.with_context = with_context
        self.add_extra_mlp = add_extra_mlp

        if activation == ActivationType.GELU:
            mlp = partial(MLP, in_features=n_embd, hidden_features=ffn_hidden, bias=bias, dropout=dropout)
        elif activation == ActivationType.FUSED_SWIGLU:
            mlp = partial(xops.SwiGLU, in_features=n_embd, hidden_features=ffn_hidden, bias=bias)
        else:
            raise NotImplementedError(f"activation type {activation} not implemented")

        self.ln_1 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
        self.attn = MultiHeadAttention(
            n_embd=n_embd, n_head=n_head, bias=bias, attention_config=attention_config, attention_type=attention_type
        )

        if not self.with_context or self.add_extra_mlp:
            self.ln_2 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
            self.mlp = mlp()

        if self.with_context:
            self.ln_3 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
            self.cross_attn = MultiHeadAttention(
                n_embd=n_embd,
                n_head=n_head,
                bias=bias,
                attention_config=attention_config,
                attention_type=AttentionType.CROSS_ATTENTION,
            )
            self.ln_4 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
            self.mlp_2 = mlp()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        if not self.with_context or self.add_extra_mlp:
            x = x + self.mlp(self.ln_2(x))
        if self.with_context:
            x = x + self.cross_attn(self.ln_3(x), context=context)
            x = x + self.mlp_2(self.ln_4(x))
        return x


class MultiModalTextDecoder(NNModel):
    def __init__(
            self,
            sample_key: str,
            prediction_key: str,
            block_size: int,
            vocab_size: int,
            n_layer: int,
            n_head: int,
            n_embd: int,
            ffn_hidden: int,
            dropout: float,
            bias: bool,
            activation: ActivationType,
            epsilon: float,
            attention_config: AttentionConfig,
    ):
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.block_size = block_size

        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList(
                    [
                        TransformerBlock(
                            n_embd=n_embd,
                            bias=bias,
                            epsilon=epsilon,
                            activation=activation,
                            n_head=n_head,
                            dropout=dropout,
                            ffn_hidden=ffn_hidden,
                            with_context=True,
                            attention_type=AttentionType.CAUSAL_SELF_ATTENTION,
                            attention_config=attention_config,
                            add_extra_mlp=False,
                        )
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon),
            )
        )
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs[self.sample_key]
        for block in self.transformer.h:
            x = block(x, context=inputs["context"])
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return {self.prediction_key: logits}
