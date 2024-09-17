from functools import partial
from typing import Dict

import torch
from torch import nn

from modalities.models.gpt2.gpt2_model import ActivationType
from modalities.models.model import NNModel, SwiGLU
from modalities.nn.attention import AttentionConfig, AttentionType, MultiHeadAttention
from modalities.nn.mlp import MLP


class TransformerBlock(nn.Module):
    """Transformer block class."""

    def __init__(
        self,
        n_embd: int,
        bias: bool,
        epsilon: float,
        activation: ActivationType,
        n_head: int,
        dropout: float,
        ffn_hidden: int,
        with_context: bool,
        attention_type: AttentionType,
        attention_config: AttentionConfig = None,
        add_extra_mlp: bool = False,
    ):
        """
        Initializes the TransformerBlock object.

        Args:
            n_embd (int): The size of the embeddings.
            bias (bool): Flag indicating whether to include bias terms.
            epsilon (float): Small value to avoid division by zero in LayerNorm.
            activation (ActivationType): The type of activation function to use.
            n_head (int): The number of attention heads.
            dropout (float): The dropout rate.
            ffn_hidden (int): The number of hidden units in the feed-forward network.
            with_context (bool): Flag indicating whether to include context in the decoder.
            attention_type (AttentionType): The type of attention mechanism to use.
            attention_config (AttentionConfig, optional): The configuration for the attention mechanism.
            Defaults to None.
            add_extra_mlp (bool, optional): Flag indicating whether to add an extra MLP layer. Defaults to False.
        """
        super().__init__()
        self.with_context = with_context
        self.add_extra_mlp = add_extra_mlp

        if activation == ActivationType.GELU:
            mlp = partial(MLP, in_features=n_embd, hidden_features=ffn_hidden, bias=bias, dropout=dropout)
        elif activation == ActivationType.SWIGLU:
            mlp = partial(SwiGLU, n_embd=n_embd, bias=bias)
        else:
            raise NotImplementedError(f"activation type {activation} not implemented")

        self.ln_1 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
        self.attn = MultiHeadAttention(
            n_embd=n_embd, n_head=n_head, bias=bias, attention_config=attention_config, attention_type=attention_type
        )

        if not self.with_context or self.add_extra_mlp:
            self.ln_2 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
            self.mlp = mlp()

        if self.with_context:
            self.ln_3 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
            self.cross_attn = MultiHeadAttention(
                n_embd=n_embd,
                n_head=n_head,
                bias=bias,
                attention_config=attention_config,
                attention_type=AttentionType.CROSS_ATTENTION,
            )
            self.ln_4 = nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon)
            self.mlp_2 = mlp()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the TransformerBlock module.

        Args:
            x (torch.Tensor): Input tensor.
            context (torch.Tensor, optional): Context tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x + self.attn(self.ln_1(x))
        if not self.with_context or self.add_extra_mlp:
            x = x + self.mlp(self.ln_2(x))
        if self.with_context:
            x = x + self.cross_attn(self.ln_3(x), context=context)
            x = x + self.mlp_2(self.ln_4(x))
        return x


class MultiModalTextDecoder(NNModel):
    """MultiModalTextDecoder class."""

    def __init__(
        self,
        sample_key: str,
        prediction_key: str,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        activation: ActivationType,
        epsilon: float,
        attention_config: AttentionConfig,
    ):
        """
        Initializes the MultiModalTextDecoder object.

        Args:
            sample_key (str): The key for the input samples.
            prediction_key (str): The key for the predictions.
            block_size (int): The size of the blocks.
            vocab_size (int): The size of the vocabulary.
            n_layer (int): The number of layers.
            n_head (int): The number of attention heads.
            n_embd (int): The dimension of the embeddings.
            ffn_hidden (int): The size of the feed-forward network hidden layer.
            dropout (float): The dropout rate.
            bias (bool): Flag indicating whether to include bias terms.
            activation (ActivationType): The activation function to use.
            epsilon (float): The epsilon value for layer normalization.
            attention_config (AttentionConfig): The attention configuration.

        Returns:
            None
        """
        super().__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.block_size = block_size

        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList(
                    [
                        TransformerBlock(
                            n_embd=n_embd,
                            bias=bias,
                            epsilon=epsilon,
                            activation=activation,
                            n_head=n_head,
                            dropout=dropout,
                            ffn_hidden=ffn_hidden,
                            with_context=True,
                            attention_type=AttentionType.CAUSAL_SELF_ATTENTION,
                            attention_config=attention_config,
                            add_extra_mlp=False,
                        )
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(normalized_shape=n_embd, bias=bias, eps=epsilon),
            )
        )
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size, bias=False)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MultiModalTextDecoder module.

        Args:
            inputs (dict[str, torch.Tensor]): Input dictionary containing the input tensors.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing the output logits tensor:
        """
        x = inputs[self.sample_key]
        for block in self.transformer.h:
            x = block(x, context=inputs["context"])
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return {self.prediction_key: logits}
