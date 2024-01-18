import math
from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 8,
        bias: bool = True,
        dropout: float = 0.0,
        use_flash: bool = True,
        is_causal: bool = False,
        use_cross_attention: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd needs to be devisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.use_flash = use_flash
        self.is_causal = is_causal
        self.use_cross_attention = use_cross_attention

        self.wq = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)
        self.wk = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)
        self.wv = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)
        self.c_proj = nn.Linear(in_features=n_embd, out_features=n_embd, bias=bias)

        if not self.use_flash:
            self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        context = context if self.use_cross_attention else x
        B, T, C = x.shape
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

    def _forward_input_projection(self, x, context):
        B, T, C = x.shape
        _, Tc, Cc = context.shape
        k = self.wk(context).view(B, Tc, self.n_head, Cc // self.n_head).transpose(1, 2)
        q = self.wq(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.wv(context).view(B, Tc, self.n_head, Cc // self.n_head).transpose(1, 2)
        return q, k, v

    def _forward_attention(self, query, key, value):
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        if self.is_causal:
            T = query.size(2)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ value
