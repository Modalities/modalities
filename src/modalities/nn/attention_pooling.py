import torch
from torch import nn

from modalities.models.gpt2.gpt2_model import LayerNorm
from modalities.nn.attention import Attention


class AttentionPooling(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool,
        epsilon: float,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)
        self.attn = Attention(n_embd, n_head, use_cross_attention=True)
        self.ln_2 = LayerNorm(ndim=n_embd, bias=bias, epsilon=epsilon)

    def forward(self, vision_embd: torch.Tensor, vision_queries: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(vision_embd)
        x = self.attn(vision_queries, context=x)
        x = self.ln_2(x)
        return x
