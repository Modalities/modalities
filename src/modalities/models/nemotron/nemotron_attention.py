"""Grouped-query causal self-attention with a head dimension decoupled from the model dimension.

This cannot reuse the GPT2 attention module: GPT2 derives the head dimension as
``n_embd // n_head_q``, whereas Nemotron-3 Nano uses 32 query heads of dimension 128 on a model
dimension of 2688 (32 * 128 = 4096, which is deliberately larger than 2688). Attention here also
applies no positional transform at all, since the Mamba layers supply positional information.
"""

import math
from enum import Enum

import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_func
except ModuleNotFoundError:
    flash_attn_func = None


class NemotronAttentionImplementation(str, Enum):
    """
    Enum of the supported attention kernels.

    Attributes:
        MANUAL (str): An explicit softmax implementation. Slow, but runs anywhere and is used as
            the reference in tests.
        PYTORCH_FLASH (str): ``torch.nn.functional.scaled_dot_product_attention``.
        DAO_FLASH (str): The ``flash-attn`` package's kernel.
    """

    MANUAL = "manual"
    PYTORCH_FLASH = "pytorch_flash"
    DAO_FLASH = "dao_flash"


class NemotronSelfAttention(nn.Module):
    """
    Causal grouped-query self-attention.

    Query, key and value projections are separate and bias-free. The number of key/value heads may
    be smaller than the number of query heads (grouped-query attention); key and value heads are
    repeated to match the queries before the attention kernel is invoked.
    """

    def __init__(
        self,
        n_embd: int,
        n_head_q: int,
        n_head_kv: int,
        head_dim: int,
        attention_implementation: NemotronAttentionImplementation = NemotronAttentionImplementation.PYTORCH_FLASH,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initializes the NemotronSelfAttention module.

        Args:
            n_embd (int): The model dimension.
            n_head_q (int): The number of query heads.
            n_head_kv (int): The number of key/value heads. Must divide ``n_head_q``.
            head_dim (int): The dimension of a single attention head. Independent of ``n_embd``.
            attention_implementation (NemotronAttentionImplementation): Which kernel to use.
            bias (bool): Whether the projections use a bias.
            dropout (float): Attention dropout probability.

        Raises:
            ValueError: If ``n_head_q`` is not divisible by ``n_head_kv``.
        """
        super().__init__()
        if n_head_q % n_head_kv != 0:
            raise ValueError(f"n_head_q ({n_head_q}) must be divisible by n_head_kv ({n_head_kv}).")

        self.n_embd = n_embd
        self.n_head_q = n_head_q
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.n_rep = n_head_q // n_head_kv
        self.dropout = dropout
        self.attention_implementation = NemotronAttentionImplementation(attention_implementation)

        self.q_attn = nn.Linear(in_features=n_embd, out_features=n_head_q * head_dim, bias=bias)
        self.k_attn = nn.Linear(in_features=n_embd, out_features=n_head_kv * head_dim, bias=bias)
        self.v_attn = nn.Linear(in_features=n_embd, out_features=n_head_kv * head_dim, bias=bias)
        self.c_proj = nn.Linear(in_features=n_head_q * head_dim, out_features=n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeats each key/value head ``n_rep`` times to match the number of query heads.

        Args:
            x (torch.Tensor): Tensor of shape ``(B, n_head_kv, T, head_dim)``.
            n_rep (int): The repetition factor.

        Returns:
            torch.Tensor: Tensor of shape ``(B, n_head_kv * n_rep, T, head_dim)``.
        """
        if n_rep == 1:
            return x
        batch_size, n_head_kv, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(batch_size, n_head_kv, n_rep, seq_len, head_dim)
            .reshape(batch_size, n_head_kv * n_rep, seq_len, head_dim)
        )

    def _execute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Runs the configured causal attention kernel.

        Args:
            q (torch.Tensor): Queries of shape ``(B, n_head_q, T, head_dim)``.
            k (torch.Tensor): Keys of shape ``(B, n_head_kv, T, head_dim)``.
            v (torch.Tensor): Values of shape ``(B, n_head_kv, T, head_dim)``.

        Raises:
            NotImplementedError: If the DAO flash attention kernel is requested but not installed.

        Returns:
            torch.Tensor: Attention output of shape ``(B, T, n_head_q, head_dim)``.
        """
        dropout_p = self.dropout if self.training else 0.0

        if self.attention_implementation == NemotronAttentionImplementation.DAO_FLASH:
            if flash_attn_func is None:
                raise NotImplementedError("Dao flash attention is requested but flash-attn is not installed.")
            # flash_attn_func handles grouped-query attention natively and wants (B, T, H, hd).
            return flash_attn_func(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                dropout_p=dropout_p,
                causal=True,
            )

        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        if self.attention_implementation == NemotronAttentionImplementation.PYTORCH_FLASH:
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            seq_len = q.size(-2)
            scale = 1.0 / math.sqrt(q.size(-1))
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
            attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = torch.dropout(attn_weights, dropout_p, train=self.training)
            y = attn_weights @ v

        return y.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input of shape ``(B, T, n_embd)``.

        Returns:
            torch.Tensor: Output of shape ``(B, T, n_embd)``.
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_attn(x).view(batch_size, seq_len, self.n_head_q, self.head_dim).transpose(1, 2)
        k = self.k_attn(x).view(batch_size, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)
        v = self.v_attn(x).view(batch_size, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)

        y = self._execute_attention(q, k, v)
        y = y.reshape(batch_size, seq_len, self.n_head_q * self.head_dim)
        return self.resid_dropout(self.c_proj(y))
