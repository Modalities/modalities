# The single-operator pre-norm residual layer structure is adapted from NVIDIA's Megatron-LM
# (megatron/core/ssm/mamba_layer.py::MambaLayer and
# megatron/core/models/hybrid/hybrid_block.py::HybridStack).
# Copyright (c) 2024-2026, NVIDIA CORPORATION. Copyright (c) 2024, Tri Dao, Albert Gu.
# Licensed under the Apache License, Version 2.0.

"""The four residual sublayer types of a hybrid Mamba-Transformer stack.

Unlike a classical transformer block, which bundles attention and a feed-forward network, every
layer here wraps exactly *one* operator in a pre-norm residual connection::

    x = x + operator(norm(x))

A model is then a sequence of such single-operator layers described by a layer pattern, e.g.
``"MEM*E"``. This is the structure of Nemotron-H and Nemotron-3 Nano, and it is what allows the
Mamba / attention / MoE ratio to be tuned independently.
"""

import torch
import torch.nn as nn

from modalities.models.components.mamba2.mamba2_mixer import Mamba2Mixer
from modalities.models.components.moe.moe import MoE
from modalities.models.nemotron.nemotron_attention import NemotronSelfAttention
from modalities.models.nemotron.nemotron_mlp import SquaredReLUMLP


class _ResidualLayer(nn.Module):
    """Base class implementing the shared pre-norm residual structure."""

    def __init__(self, norm: nn.Module):
        """
        Initializes the residual layer.

        Args:
            norm (nn.Module): The pre-normalization module, owned exclusively by this layer.
        """
        super().__init__()
        self.norm = norm

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the layer's operator to the normalized input.

        Args:
            x (torch.Tensor): The normalized input of shape ``(B, L, n_embd)``.

        Returns:
            torch.Tensor: The operator output of shape ``(B, L, n_embd)``.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: a pre-norm residual application of the layer's operator.

        Args:
            x (torch.Tensor): Input of shape ``(B, L, n_embd)``.

        Returns:
            torch.Tensor: Output of shape ``(B, L, n_embd)``.
        """
        return x + self._operator(self.norm(x))


class Mamba2Layer(_ResidualLayer):
    """A residual layer whose operator is a Mamba-2 mixer."""

    def __init__(self, norm: nn.Module, mixer: Mamba2Mixer):
        """
        Initializes the Mamba2Layer.

        Args:
            norm (nn.Module): The pre-normalization module.
            mixer (Mamba2Mixer): The Mamba-2 mixer.
        """
        super().__init__(norm=norm)
        self.mixer = mixer

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class NemotronAttentionLayer(_ResidualLayer):
    """A residual layer whose operator is grouped-query causal self-attention."""

    def __init__(self, norm: nn.Module, attn: NemotronSelfAttention):
        """
        Initializes the NemotronAttentionLayer.

        Args:
            norm (nn.Module): The pre-normalization module.
            attn (NemotronSelfAttention): The self-attention module.
        """
        super().__init__(norm=norm)
        self.attn = attn

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class NemotronMoELayer(_ResidualLayer):
    """A residual layer whose operator is a sparse mixture-of-experts feed-forward block."""

    def __init__(self, norm: nn.Module, moe: MoE):
        """
        Initializes the NemotronMoELayer.

        Args:
            norm (nn.Module): The pre-normalization module.
            moe (MoE): The mixture-of-experts block.
        """
        super().__init__(norm=norm)
        self.moe = moe

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x)


class NemotronMLPLayer(_ResidualLayer):
    """A residual layer whose operator is a dense squared-ReLU feed-forward network."""

    def __init__(self, norm: nn.Module, mlp: SquaredReLUMLP):
        """
        Initializes the NemotronMLPLayer.

        Args:
            norm (nn.Module): The pre-normalization module.
            mlp (SquaredReLUMLP): The dense feed-forward network.
        """
        super().__init__(norm=norm)
        self.mlp = mlp

    def _operator(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
