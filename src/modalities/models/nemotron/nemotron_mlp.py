# The squared ReLU activation follows NVIDIA's Megatron-LM (megatron/core/activations.py),
# Copyright (c) NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0.

"""Non-gated squared-ReLU feed-forward network.

Nemotron models use ``ReLU(x)^2`` rather than a gated activation such as SwiGLU. This halves the
number of matrices per feed-forward block (no gate projection), which is what makes a granular
128-expert MoE affordable.
"""

import torch
import torch.nn as nn

from modalities.models.components.moe.experts import squared_relu


class SquaredReLUMLP(nn.Module):
    """
    A two-layer feed-forward network with a squared ReLU activation.

    Used both for dense feed-forward layers and as the body of the shared expert in an MoE layer.
    """

    def __init__(self, n_embd: int, ffn_hidden: int, bias: bool = False):
        """
        Initializes the SquaredReLUMLP.

        Args:
            n_embd (int): The model dimension.
            ffn_hidden (int): The hidden dimension.
            bias (bool): Whether the projections use a bias. Nemotron uses no biases anywhere.
        """
        super().__init__()
        self.c_fc = nn.Linear(in_features=n_embd, out_features=ffn_hidden, bias=bias)
        self.c_proj = nn.Linear(in_features=ffn_hidden, out_features=n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SquaredReLUMLP.

        Args:
            x (torch.Tensor): Input of shape ``(..., n_embd)``.

        Returns:
            torch.Tensor: Output of shape ``(..., n_embd)``.
        """
        return self.c_proj(squared_relu(self.c_fc(x)))
