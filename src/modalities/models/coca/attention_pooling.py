import torch
from torch import nn

from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention


class AttentionPooling(nn.Module):
    """Attention pooling class."""

    def __init__(self, n_embd: int, n_head: int, bias: bool, epsilon: float, attention_config: AttentionConfig = None):
        """
        Initializes an instance of the AttentionPooling class.

        Args:
            n_embd (int): The size of the embeddings.

            n_head (int): The number of attention heads.
            bias (bool): Flag indicating whether to include bias in the layer normalization.
            epsilon (float): A small value to avoid division by zero in layer normalization.
            attention_config (AttentionConfig, optional): The configuration for attention mechanism. Defaults to None.

        Returns:
            None
        """
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
        """
        Forward pass of the attention pooling module.

        Args:
            queries (torch.Tensor): The input queries tensor.
            context (torch.Tensor): The input context tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.ln_1(context)
        x = self.attn(queries, context=x)
        x = self.ln_2(x)
        return x
