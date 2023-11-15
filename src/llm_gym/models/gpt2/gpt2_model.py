import math
from abc import abstractmethod
from enum import Enum
from typing import Dict, Optional
from llm_gym.models.model import NNModel

import torch
import torch.nn as nn
import xformers.ops as xops
from pydantic import BaseModel, confloat, conint
from torch.nn import functional as F


# GPT2 implementation taken from nanogpt https://github.com/karpathy/nanoGPT


class Attention(Enum):
    DEFAULT_ATTENTION = "default_attention"
    PYTORCH_FLASH_ATTENTION = "pytorch_flash_attention"


class Activation(Enum):
    GELU = "gelu"
    FUSED_SWIGLU = "fused_swiglu"


class AttentionConfig(BaseModel):
    attention_type: Attention
    scaling_factor: conint(ge=1)


class GPTConfig(BaseModel):
    sample_key: str
    prediction_key: str
    block_size: conint(ge=1)
    vocab_size: conint(ge=1)  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: conint(ge=1)
    n_head: conint(ge=1)
    n_embd: conint(ge=1)
    dropout: confloat(ge=0.0)
    bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention: AttentionConfig
    activation: Activation
    epsilon: confloat(ge=0.0)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input=input,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.epsilon,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            in_features=config.n_embd,
            out_features=config.attention.scaling_factor * config.n_embd,
            bias=config.bias,
        )

        # output projection
        self.c_proj = nn.Linear(
            in_features=config.n_embd,
            out_features=config.n_embd,
            bias=config.bias,
        )

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = config.attention.attention_type == Attention.PYTORCH_FLASH_ATTENTION

        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerMLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(
            in_features=config.n_embd,
            out_features=4 * config.n_embd,
            bias=config.bias,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            in_features=4 * config.n_embd,
            out_features=config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=config.n_embd, bias=config.bias, epsilon=config.epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(ndim=config.n_embd, bias=config.bias, epsilon=config.epsilon)

        if config.activation == Activation.GELU:
            self.mlp = TransformerMLP(config)
        elif config.activation == Activation.FUSED_SWIGLU:
            hidden_dim = 256 * ((int(2 * 4 * config.n_embd / 3) + 256 - 1) // 256)
            self.mlp = xops.SwiGLU(config.n_embd, hidden_dim, config.n_embd, bias=False)
        else:
            raise Exception("unimplemented activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LLM(NNModel):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.sample_key = config.sample_key
        self.prediction_key = config.prediction_key

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
                wpe=nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(ndim=config.n_embd, bias=config.bias, epsilon=config.epsilon),
            )
        )
        self.lm_head = nn.Linear(in_features=config.n_embd, out_features=config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_impl(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = inputs[self.sample_key]
        device = input_ids.device
        b, t = input_ids.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return {self.prediction_key: logits}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward_impl(inputs)
