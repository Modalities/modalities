import math
from enum import Enum
from functools import partial
from typing import Annotated, Dict

import torch
import torch.nn as nn
import xformers.ops as xops
from pydantic import BaseModel, Field, model_validator
from torch.nn import functional as F

from modalities.models.model import NNModel

# GPT2 implementation taken from nanogpt https://github.com/karpathy/nanoGPT


class AttentionType(str, Enum):
    DEFAULT_ATTENTION = "default_attention"
    PYTORCH_FLASH_ATTENTION = "pytorch_flash_attention"


class ActivationType(str, Enum):
    GELU = "gelu"
    FUSED_SWIGLU = "fused_swiglu"


class WeightInitailizationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)]


class GPT2LLMConfig(BaseModel):
    sample_key: str
    prediction_key: str
    block_size: Annotated[int, Field(strict=True, ge=1)]
    vocab_size: Annotated[
        int, Field(strict=True, ge=1)
    ]  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: Annotated[int, Field(strict=True, ge=1)]
    n_head_q: Annotated[int, Field(strict=True, ge=1)]
    n_head_kv: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    ffn_hidden: Annotated[int, Field(strict=True, ge=1)]

    dropout: Annotated[float, Field(strict=True, ge=0.0)]
    bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention_type: AttentionType
    activation: ActivationType
    epsilon: Annotated[float, Field(strict=True, ge=0.0)]
    weight_init: WeightInitailizationConfig

    @model_validator(mode="after")
    def validate_sizes(self) -> "GPT2LLMConfig":
        for param, param_name in zip(
            [self.ffn_hidden, self.vocab_size, self.n_embd], ["ffn_hidden", "vocab_size", "n_embd"]
        ):
            if param % 128 != 0:
                # See https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
                raise ValueError(f"{param_name} with value {param} should be divisible by 128 for efficient training.")
        return self


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
    def __init__(
        self,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        attention_type: AttentionType,
        bias: bool,
        dropout: float,
        block_size: int,
    ):
        super().__init__()
        assert n_embd % n_head_q == 0, (
            "Embeddings get passed to `n_head_q` different heads "
            "and their dimension needs to be divisible by `n_head_q`."
        )
        assert n_head_q % n_head_kv == 0, (
            "It is necessary to have `n_head_q` divisible by `n_head_kv`."
            ' For more details, read about "Grouped Query Attention"'
        )

        self.n_rep = n_head_q // n_head_kv

        # query, key, value projections (separate)
        self.q_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd,
            bias=bias,
        )
        self.k_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd // self.n_rep,
            bias=bias,
        )
        self.v_attn = nn.Linear(
            in_features=n_embd,
            out_features=n_embd // self.n_rep,
            bias=bias,
        )

        # output projection
        self.c_proj = nn.Linear(
            in_features=n_embd,
            out_features=n_embd,
            bias=bias,
        )

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head_q = n_head_q
        self.n_head_kv = n_head_kv

        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = attention_type == AttentionType.PYTORCH_FLASH_ATTENTION

        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()  # batch size (B), sequence length (T), embedding dimensionality (self.n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_attn(x)  # (B, T, n_embd)
        k = self.k_attn(x)  # (B, T, n_embd / n_rep)
        v = self.v_attn(x)  # (B, T, n_embd / n_rep)

        q = q.view(B, T, self.n_head_q, self.n_embd // self.n_head_q).transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.view(B, T, self.n_head_kv, self.n_embd // self.n_head_q).transpose(1, 2)  # (B, nh_kv, T, hs)
        v = v.view(B, T, self.n_head_kv, self.n_embd // self.n_head_q).transpose(1, 2)  # (B, nh_kv, T, hs)

        # repeat k/v heads if self.n_rep > 1
        k = repeat_kv(k, self.n_rep)  # (B, nh_q, T, hs)
        v = repeat_kv(v, self.n_rep)  # (B, nh_q, T, hs)

        # causal self-attention; Self-attend: (B, nh_q, T, hs) x (B, nh_q, hs, T) -> (B, nh_q, T, T)
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
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh_q, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh_q, T, T) x (B, nh_q, T, hs) -> (B, nh_q, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        )  # (B, T, n_embd), re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))  # (B, T, n_embd)
        return y


class TransformerMLP(nn.Module):
    def __init__(self, n_embd: int, ffn_hidden: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(
            in_features=n_embd,
            out_features=ffn_hidden,  # 4 * n_embd,
            bias=bias,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            in_features=ffn_hidden,
            out_features=n_embd,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        bias: bool,
        epsilon: float,
        activation: ActivationType,
        n_head_q: int,
        n_head_kv: int,
        attention_type: AttentionType,
        dropout: float,
        block_size: int,
        ffn_hidden: int,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)
        self.attn = CausalSelfAttention(
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            attention_type=attention_type,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
        )
        self.ln_2 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)

        if activation == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation == ActivationType.FUSED_SWIGLU:
            hidden_dim = 256 * ((int(2 * 4 * n_embd / 3) + 256 - 1) // 256)
            self.mlp = xops.SwiGLU(n_embd, hidden_dim, n_embd, bias=False)
        else:
            raise Exception("unimplemented activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LLM(NNModel):
    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        attention_type: AttentionType,
        activation: ActivationType,
        epsilon: float,
        weight_init: WeightInitailizationConfig,
    ):
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.block_size = block_size

        assert vocab_size is not None
        assert block_size is not None

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
                            n_head_q=n_head_q,
                            n_head_kv=n_head_kv,
                            attention_type=attention_type,
                            dropout=dropout,
                            block_size=block_size,
                            ffn_hidden=ffn_hidden,
                        )
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon),
            )
        )
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

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

    def forward_impl(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = inputs[self.sample_key]
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Source code adopted from
      https://github.com/facebookresearch/llama/blob/9a001c7a0987afd7b8de94e538916eff8950a73a/llama/model.py#L164
    Adapted ordered dimensions and namings: bs=B, n_kv_heads=nh_kv, slen=T, head_dim=hs
    """
    B, nh_kv, T, hs = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(B, nh_kv, n_rep, T, hs).reshape(B, nh_kv * n_rep, T, hs)
