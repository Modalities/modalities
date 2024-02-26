import math
from functools import partial
from typing import Dict

import torch
import xformers.ops as xops
from torch import nn

from modalities.models.gpt2.gpt2_model import (
    ActivationType,
    AttentionConfig,
    LayerNorm,
    TransformerMLP,
    WeightInitailizationConfig,
)
from modalities.models.model import NNModel
from modalities.nn.attention import Attention


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
        self.attn = Attention(n_embd, n_head, bias, is_causal=False, use_cross_attention=True)
        self.ln_2 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)

        if activation == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation == ActivationType.FUSED_SWIGLU:
            self.mlp = xops.SwiGLU(n_embd, ffn_hidden, n_embd, bias=False)
        else:
            raise Exception("unimplemented activation")

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), context=context)
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiModalDecoder(NNModel):
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
        attention: AttentionConfig,
        activation: ActivationType,
        epsilon: float,
        weight_init: WeightInitailizationConfig,
    ):
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.block_size = block_size

        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList(
                    [
                        MultiModalBlock(
                            n_embd=n_embd,
                            bias=bias,
                            epsilon=epsilon,
                            activation=activation,
                            n_head=n_head,
                            dropout=dropout,
                            ffn_hidden=ffn_hidden,
                        )
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon),
            )
        )
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False)

        # init all weights
        self.apply(partial(self._init_weights, weight_init=weight_init))
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=weight_init.mean, std=weight_init.std / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module, weight_init: WeightInitailizationConfig):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=weight_init.mean, std=weight_init.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=weight_init.mean, std=weight_init.std)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs[self.sample_key]
        for block in self.transformer.h:
            x = block(x, context=inputs["context"])
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return {self.prediction_key: logits}
