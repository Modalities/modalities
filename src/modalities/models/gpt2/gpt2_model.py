import math
from copy import deepcopy
from enum import Enum
from typing import Annotated, Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_func
except ModuleNotFoundError:
    flash_attn_func = None

from pydantic import BaseModel, Field, model_validator, validator

from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.config.utils import convert_base_model_config_to_dict
from modalities.models.model import ActivationType, NNModel, SwiGLU
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


class AttentionImplementation(str, Enum):
    MANUAL = "manual"
    PYTORCH_FLASH = "pytorch_flash"
    DAO_FLASH = "dao_flash"


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

    qkv_transforms: List[QueryKeyValueTransformConfig]


class GPT2LLMConfig(BaseModel):
    sample_key: str
    prediction_key: str
    poe_type: PositionTypes
    sequence_length: Annotated[int, Field(strict=True, ge=1)]
    vocab_size: Annotated[
        int, Field(strict=True, ge=1)
    ]  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: Annotated[int, Field(strict=True, ge=1)]
    n_head_q: Annotated[int, Field(strict=True, ge=1)]
    n_head_kv: Annotated[int, Field(strict=True, ge=1)]
    n_embd: Annotated[int, Field(strict=True, ge=1)]
    ffn_hidden: Annotated[int, Field(strict=True, ge=1)]
    dropout: Annotated[float, Field(strict=True, ge=0.0)]
    bias: bool  # True: bias in Linears like GPT-2. False: a bit better and faster
    attention_config: AttentionConfig
    attention_implementation: AttentionImplementation
    activation_type: ActivationType
    attention_norm: PydanticPytorchModuleType
    ffn_norm: PydanticPytorchModuleType
    lm_head_norm: PydanticPytorchModuleType

    @model_validator(mode="after")
    def check_divisibility(self) -> "GPT2LLMConfig":
        if self.n_head_q % self.n_head_kv != 0:
            raise ValueError("n_head_q must be divisible by n_head_kv")
        return self

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
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        attention_config: AttentionConfig,
        attention_impl: AttentionImplementation,
        bias: bool,
        dropout: float,
    ):
        super().__init__()
        assert n_embd % n_head_q == 0, "`n_embd needs` to be divisible by `n_head_q`."
        assert n_head_q % n_head_kv == 0, "`n_head_q needs` to be divisible by `n_head_kv`."

        self.n_rep = n_head_q // n_head_kv
        self.attention_impl = attention_impl

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
        self.n_head_q = n_head_q
        self.n_head_kv = n_head_kv

        self.n_embd = n_embd
        # TODO: we might want different values for attention_dropout and linear_dropout
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(self.dropout)

        # TODO: inject QKVTransforms from outside
        self.qkv_transforms = nn.ModuleList(
            transform_config.type_hint.value(
                **convert_base_model_config_to_dict(transform_config.config)
            )  # TODO refactor, still uses the legacy type_hint
            for transform_config in attention_config.qkv_transforms
        )

    def projection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        return self.q_attn(x), self.k_attn(x), self.v_attn(x)

    @staticmethod
    def execute_qkv_transforms(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qkv_transforms: nn.ModuleList, n_head_q: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, embedding_dim = q.size()
        # hidden dimension of single head
        # Note, that number of heads does not change the overall parameters of the networks
        # to scale up the network we either have to increase the embedding_dim or the number of layers
        n_head_dim = embedding_dim // n_head_q

        q = q.view(batch_size, sequence_length, n_head_q, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_q, T, hd)
        k = k.view(batch_size, sequence_length, -1, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_kv, T, hd)
        v = v.view(batch_size, sequence_length, -1, n_head_dim).transpose(1, 2).contiguous()  # (B, nh_kv, T, hd)

        for transform in qkv_transforms:
            q, k, v = transform(q, k, v)

        return q, k, v

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Source code adopted from
        https://github.com/facebookresearch/llama/blob/9a001c7a0987afd7b8de94e538916eff8950a73a/llama/model.py#L164
        Adapted ordered dimensions and namings: bs=B, n_kv_heads=nh_kv, slen=T, head_dim=hs
        """
        B, nh_kv, T, hs = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(B, nh_kv, n_rep, T, hs).reshape(B, nh_kv * n_rep, T, hs)

    @classmethod
    def repeat_kv_heads(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # repeat k/v heads if self.n_rep > 1
        n_head_q = q.shape[1]
        n_head_kv = k.shape[1]
        if n_head_q != n_head_kv:
            n_rep = n_head_q // n_head_kv
            k = cls._repeat_kv(k, n_rep)  # (B, nh_q, T, hs)
            v = cls._repeat_kv(v, n_rep)  # (B, nh_q, T, hs)
        return k, v

    @classmethod
    def execute_attention(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout: float,
        attention_impl: AttentionImplementation,
    ) -> torch.Tensor:
        if attention_impl == AttentionImplementation.MANUAL:
            k, v = cls.repeat_kv_heads(q, k, v)  # for GQA (group query attention)
            y = manual_scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=dropout,
                is_causal=True,
            )  # (B, nh_q, T, hd)
            y = y.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
        elif attention_impl == AttentionImplementation.PYTORCH_FLASH:
            k, v = cls.repeat_kv_heads(q, k, v)  # for GQA (group query attention)
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=dropout,
                is_causal=True,
            )  # (B, nh_q, T, hd)
            y = y.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
        elif attention_impl == AttentionImplementation.DAO_FLASH:
            # Due to the lack of GPUs in github actions and the requirement of those in the flash-attn library,
            # we have to check if the library is installed and raise an error if not.
            # Note, that the library is not required for the CPU-only tests.
            if flash_attn_func is None:
                raise NotImplementedError("ERROR! Dao Flash Attention is not installed.")
            # the next three lines are only needed for flash-attn from Daio Lab
            q = q.transpose(1, 2).contiguous()  # (B, T, nh_q, hd)
            k = k.transpose(1, 2).contiguous()  # (B, T, nh_kv, hd)
            v = v.transpose(1, 2).contiguous()  # (B, T, nh_kv, hd)
            y = flash_attn_func(
                q, k, v, dropout_p=dropout, causal=True, softmax_scale=None, window_size=(-1, -1)
            )  # (B, T, nh_q, hd)
        else:
            raise NotImplementedError(f"Attention implementation {attention_impl} not supported")
        return y  # (B, T, nh_q, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()  # batch size (B), sequence length (T), embedding dimensionality (self.n_embd)
        q, k, v = self.projection(x)  # q: (B, T, n_embd), k: (B, T, n_embd // n_rep), v: (B, T, n_embd // n_rep)

        # q: (B, nh_q, T, hd), k: (B, nh_kv, T, hd), v: (B, nh_kv, T, hd)
        q, k, v = CausalSelfAttention.execute_qkv_transforms(q, k, v, self.qkv_transforms, self.n_head_q)
        y = CausalSelfAttention.execute_attention(q, k, v, self.dropout, self.attention_impl)  # (B, T, nh_q, hd)
        y = y.reshape(B, T, self.n_embd)  # (B, T, n_embd), re-assemble all head outputs side by side
        return self.resid_dropout(self.c_proj(y))  # (B, T, n_embd), output projection


class TransformerMLP(nn.Module):
    def __init__(self, n_embd: int, ffn_hidden: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(
            in_features=n_embd,
            out_features=ffn_hidden,  # best practice: 4 * n_embd,
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
        n_head_q: int,
        n_head_kv: int,
        activation_type: ActivationType,
        attention_impl: AttentionImplementation,
        attention_config: AttentionConfig,
        dropout: float,
        ffn_hidden: int,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.ffn_norm = ffn_norm
        self.attn = CausalSelfAttention(
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            attention_config=attention_config,
            attention_impl=attention_impl,
            bias=bias,
            dropout=dropout,
        )
        if activation_type == ActivationType.GELU:
            self.mlp = TransformerMLP(n_embd=n_embd, ffn_hidden=ffn_hidden, bias=bias, dropout=dropout)
        elif activation_type == ActivationType.SWIGLU:
            self.mlp = SwiGLU(n_embd=n_embd, bias=bias)
        else:
            raise NotImplementedError("unimplemented activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attention_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class GPT2LLM(NNModel):
    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        poe_type: PositionTypes,
        sequence_length: int,
        vocab_size: int,
        n_layer: int,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        activation_type: ActivationType,
        attention_implementation: AttentionImplementation,
        attention_config: AttentionConfig,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
        lm_head_norm: nn.Module,
        seed: int = None,
    ):
        weight_decay_groups = {
            "linear": [".attn", ".mlp"],
            "embedding": [".wte", ".wpe"],
            "layernorm": [".attention_norm", ".ffn_norm", ".lm_head_norm"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups, seed=seed)
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.sequence_length = sequence_length
        self.poe_type = poe_type

        assert vocab_size is not None
        assert sequence_length is not None

        # TODO: dependency injection
        if poe_type is PositionTypes.ABSOLUTE:
            wpe = nn.Embedding(num_embeddings=sequence_length, embedding_dim=n_embd)
        elif poe_type is PositionTypes.NOPE:
            # Using a pre-trained layer, requires to define a separate FSDP unit for the frozen layer c.f.
            # https://github.com/huggingface/accelerate/issues/807
            # wpe = nn.Embedding.from_pretrained(torch.zeros(sequence_length, n_embd))
            wpe = nn.Identity()
        else:
            raise TypeError(f"{poe_type} not supported")

        if poe_type is not PositionTypes.NOPE and RotaryTransform in [
            config.type_hint.value for config in attention_config.qkv_transforms
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
                            n_head_q=n_head_q,
                            n_head_kv=n_head_kv,
                            activation_type=activation_type,
                            attention_impl=attention_implementation,
                            attention_config=attention_config,
                            dropout=dropout,
                            ffn_hidden=ffn_hidden,
                            attention_norm=deepcopy(attention_norm),
                            ffn_norm=deepcopy(ffn_norm),
                        )
                        for _ in range(n_layer)
                    ]
                ),
                lm_head_norm=lm_head_norm,
            )
        )
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

    def forward_impl(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = inputs[self.sample_key]
        device = input_ids.device
        _, t = input_ids.size()  # batch size, sequence length
        assert t <= self.sequence_length, f"Cannot forward sequence of length {t}, the model's maximum "
        f"input sequence length is only {self.sequence_length}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)

        if self.poe_type is PositionTypes.ABSOLUTE:
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            tok_emb = tok_emb + pos_emb

        # TODO: use drop out also without absolute position embedding?
        x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.lm_head_norm(x)
        logits = self.lm_head(x)
        return {self.prediction_key: logits}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward_impl(inputs)


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    """
    taken from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(
        L, S, dtype=query.dtype, device=query.device
    )  # device added (not part of the original code)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)  # device added
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
