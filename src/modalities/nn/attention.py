import math
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor, nn


class AttentionEngineType(str, Enum):
    DEFAULT_ATTENTION = "default_attention"
    PYTORCH_FLASH_ATTENTION = "pytorch_flash_attention"


class AttentionType(str, Enum):
    CAUSAL_SELF_ATTENTION = "causal_self_attention"
    NON_CAUSAL_SELF_ATTENTION = "non_causal_self_attention"
    CROSS_ATTENTION = "cross_attention"


class AttentionConfig(BaseModel):
    attention_engine_type: AttentionEngineType


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        attention_config: AttentionConfig = None,
        attention_type: AttentionType = AttentionType.CAUSAL_SELF_ATTENTION,
        n_embd: int = 768,
        n_head: int = 8,
        bias: bool = True,
        dropout: float = 0.0,
        block_size: int = 1024,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd needs to be divisible by n_head")
        if attention_config is None:
            attention_config = AttentionConfig(attention_engine_type=AttentionEngineType.DEFAULT_ATTENTION)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.use_flash = attention_config.attention_engine_type == AttentionEngineType.PYTORCH_FLASH_ATTENTION
        self.is_causal = attention_type == AttentionType.CAUSAL_SELF_ATTENTION
        self.use_cross_attention = attention_type == AttentionType.CROSS_ATTENTION

        self.wq = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)
        self.wk = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)
        self.wv = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)
        self.c_proj = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)

        if not self.use_flash:
            self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        context = context if self.use_cross_attention else x
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self._forward_input_projection(x, context=context)
        if self.use_flash:
            y = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.is_causal,
            )
        else:
            y = self._forward_attention(query=q, key=k, value=v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def _forward_input_projection(self, x: Tensor, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality (n_embd)
        _, Tc, Cc = context.shape  # batch size, context length, context embedding dimensionality
        # Note that the context length (Tc), sequence length (T) and embedding dimensionalities (C and Cc)
        # are the same for self-attention and can only differ for cross-attention
        q = self.wq(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.wk(context).view(B, Tc, self.n_head, Cc // self.n_head).transpose(1, 2)
        v = self.wv(context).view(B, Tc, self.n_head, Cc // self.n_head).transpose(1, 2)
        return q, k, v

    def _forward_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        if self.is_causal:
            T = query.size(2)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ value
