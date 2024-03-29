import torch
from torch import nn

from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention


class AttentionPooling(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool, epsilon: float, attention_config: AttentionConfig = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
        self.attn = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            attention_config=attention_config,
            attention_type=AttentionType.CROSS_ATTENTION,
        )
        self.ln_2 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(context)
        x = self.attn(queries, context=x)
        x = self.ln_2(x)
        return x
