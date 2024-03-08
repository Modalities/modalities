import torch
from torch import nn

from modalities.models.gpt2.gpt2_model import LayerNorm
from modalities.nn.attention import AttentionConfig, AttentionType, MultiheadAttention


class AttentionPooling(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool, epsilon: float, attention_config: AttentionConfig):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)
        self.attn = MultiheadAttention(
            n_embd, n_head, attention_config=attention_config, attention_type=AttentionType.CROSS_ATTENTION
        )
        self.ln_2 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(context)
        x = self.attn(queries, context=x)
        x = self.ln_2(x)
        return x
