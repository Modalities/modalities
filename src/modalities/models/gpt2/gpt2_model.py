import math
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Annotated, Dict, List, Tuple

import torch
import torch.nn as nn
import xformers.ops as xops
from pydantic import BaseModel, Field, model_validator, validator
from torch.nn import functional as F

from modalities.config.config import PydanticPytorchModuleType
from modalities.config.utils import convert_base_model_config_to_dict
from modalities.models.model import NNModel
from modalities.util import parse_enum_by_name

# GPT2 implementation taken from nanogpt https://github.com/karpathy/nanoGPT


class PositionTypes(str, Enum):
    ABSOLUTE = "ABSOLUTE"
    NOPE = "NOPE"


class QueryKeyValueTransform(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


class IdentityTransform(QueryKeyValueTransform):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return q, k, v


class RotaryTransform(QueryKeyValueTransform):
    """Implementation of Rotary Positioanl Embeddings
    Source: https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
    We added the corresponding code here, becauase there is a conflict with "@torch.jit.script" used in the
    XFormers implementation and removed in this implementation.
    """

    def __init__(self, n_embd: int, n_head: int, seq_length_dim: int = -2):
        super().__init__()
        dim_model = n_embd // n_head
        self.seq_length_dim = seq_length_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _update_cos_sin_tables(self, x):
        seq_len = x.shape[self.seq_length_dim]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_length_dim], device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb(self, x, cos, sin):
        # NOTE: This could probably be moved to Triton

        # Handle a possible sequence length mismatch in between q and k
        cos = cos[:, :, : x.shape[self.seq_length_dim], :]
        sin = sin[:, :, : x.shape[self.seq_length_dim], :]

        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k)
        q = self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached)
        k = self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)

        return q, k, v


class QueryKeyValueTransformType(Enum):
    IdentityTransform = IdentityTransform
    RotaryTransform = RotaryTransform


class AttentionType(str, Enum):
    DEFAULT_ATTENTION = "default_attention"
    PYTORCH_FLASH_ATTENTION = "pytorch_flash_attention"


class ActivationType(str, Enum):
    GELU = "gelu"
    FUSED_SWIGLU = "fused_swiglu"


class AttentionConfig(BaseModel):
    class QueryKeyValueTransformConfig(BaseModel):
        class IdentityTransformConfig(BaseModel):
            pass

        class RotaryTransformConfig(BaseModel):
            n_embd: Annotated[int, Field(strict=True, ge=0)]
            n_head: Annotated[int, Field(strict=True, ge=0)]
            seq_length_dim: Annotated[int, Field(strict=True)]

        @validator("type_hint", pre=True, always=True)
        def parse_sharding_strategy_by_name(cls, name):
            return parse_enum_by_name(name=name, enum_type=QueryKeyValueTransformType)

        type_hint: QueryKeyValueTransformType
        config: RotaryTransformConfig | IdentityTransformConfig

    attention_type: AttentionType
    qkv_transforms: List[QueryKeyValueTransformConfig]
    scaling_factor: Annotated[int, Field(strict=True, ge=1)]


class WeightInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)]


class GPT2LLMConfig(BaseModel):
    sample_key: str
    prediction_key: str
    poe_type: PositionTypes
    block_size: Annotated[int, Field(strict=True, ge=1)]
    vocab_size: Annotated[
        int, Field(strict=True, ge=1)
    ]  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: Annotated[int, Field(strict=True, ge=1)]
    n_head: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    ffn_hidden: Annotated[int, Field(strict=True, ge=1)]
    dropout: Annotated[float, Field(strict=True, ge=0.0)]
    bias: bool  # True: bias in Linears like GPT-2. False: a bit better and faster
    attention_config: AttentionConfig
    activation_type: ActivationType
    weight_init: WeightInitailizationConfig
    attention_norm: PydanticPytorchModuleType
    ffn_norm: PydanticPytorchModuleType
    lm_head_norm: PydanticPytorchModuleType
    weight_init: WeightInitializationConfig


    @model_validator(mode="after")
    def validate_sizes(self) -> "GPT2LLMConfig":
        for param, param_name in zip(
            [self.ffn_hidden, self.vocab_size, self.n_embd], ["ffn_hidden", "vocab_size", "n_embd"]
        ):
            if param % 128 != 0:
                # See https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
                raise ValueError(f"{param_name} with value {param} should be divisible by 128 for efficient training.")
        return self


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        attention_config: AttentionConfig,
        bias: bool,
        dropout: float,
        block_size: int,

    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            in_features=n_embd,
            # 3, because we have queries, keys, and values
            out_features=3 * n_embd,
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
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = attention_config.attention_type == AttentionType.PYTORCH_FLASH_ATTENTION

        # TODO: inject QKVTransforms from outside
        self.qkv_transforms = nn.ModuleList(
            transform_config.type_hint.value(**convert_base_model_config_to_dict(transform_config.config))
            for transform_config in attention.qkv_transforms
        )

        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # TODO: move logic into a function
        for qkv_transform in self.qkv_transforms:
            q, k, v = qkv_transform(q, k, v)

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
        activation_type: ActivationType,
        n_head: int,
        attention_config: AttentionConfig,
        dropout: float,
        block_size: int,
        ffn_hidden: int,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self.attn = CausalSelfAttention(
            n_head=n_head,
            n_embd=n_embd,
            attention_config=attention_config,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
        )
        if activation_type == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation_type == ActivationType.FUSED_SWIGLU:
            hidden_dim = 256 * ((int(2 * 4 * n_embd / 3) + 256 - 1) // 256)
            self.mlp = xops.SwiGLU(n_embd, hidden_dim, n_embd, bias=False)
        else:
            raise NotImplementedError("unimplemented activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_norm(x)
        x = x + self.attn(x)
        x = self.ffn_norm(x)
        x = x + self.mlp(x)
        return x


class GPT2LLM(NNModel):
    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        poe_type: PositionTypes,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        attention_config: AttentionConfig,
        activation_type: ActivationType,
        weight_init: WeightInitializationConfig,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
        lm_head_norm: nn.Module,
    ):
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.block_size = block_size
        self.poe_type = poe_type

        assert vocab_size is not None
        assert block_size is not None

        # TODO: dependency injection
        if poe_type is PositionTypes.ABSOLUTE:
            wpe = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        elif poe_type is PositionTypes.NOPE:
            # Using a pre-trained layer, requires to define a separate FSDP unit for the frozen layer c.f.
            # https://github.com/huggingface/accelerate/issues/807
            # wpe = nn.Embedding.from_pretrained(torch.zeros(block_size, n_embd))
            wpe = nn.Identity()
        else:
            raise TypeError(f"{poe_type} not supported")

        if poe_type is not PositionTypes.NOPE and RotaryTransform in [
            config.type_hint.value for config in attention.qkv_transforms
        ]:
            raise ValueError('It is expected to use "RotaryTransform" together with "NOPE".')

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd),
                wpe=wpe,
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        GPT2Block(
                            n_embd=n_embd,
                            bias=bias,
                            activation_type=activation_type,
                            n_head=n_head,
                            attention_config=attention_config,
                            dropout=dropout,
                            block_size=block_size,
                            ffn_hidden=ffn_hidden,
                            attention_norm=deepcopy(attention_norm),
                            ffn_norm=deepcopy(ffn_norm),
                        )
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=lm_head_norm,
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

    def _init_weights(self, module: nn.Module, weight_init: WeightInitializationConfig):
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

        if self.poe_type is PositionTypes.ABSOLUTE:
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            tok_emb = tok_emb + pos_emb

        # TODO: use drop out also without absolute position embedding?
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return {self.prediction_key: logits}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward_impl(inputs)
