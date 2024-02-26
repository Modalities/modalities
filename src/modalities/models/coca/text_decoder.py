import math
from functools import partial
from typing import Dict

import torch
from torch import nn

from modalities.models.gpt2.gpt2_model import ActivationType, AttentionConfig, GPT2Block, WeightInitailizationConfig
from modalities.models.model import NNModel


class TextDecoder(NNModel):
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

        self.cls_token = nn.Parameter(torch.empty(1, 1, n_embd))
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd),
                wpe=nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        GPT2Block(
                            n_embd=n_embd,
                            bias=bias,
                            epsilon=epsilon,
                            activation=activation,
                            n_head=n_head,
                            attention=attention,
                            dropout=dropout,
                            block_size=block_size,
                            ffn_hidden=ffn_hidden,
                        )
                        for _ in range(n_layer)
                    ]
                ),
            )
        )

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
        input_ids = inputs[self.sample_key]
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t + 1, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(input_ids)
        tok_emb = torch.cat([tok_emb, self.cls_token.repeat(b, 1, 1)], dim=1)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        return {self.prediction_key: x}
