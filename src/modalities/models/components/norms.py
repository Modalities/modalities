"""Normalization wrapper config shared by models outside of GPT2.

GPT2 carries its own, historically grown ``LayerNormWrapperConfig``. This module provides the
same declarative "which norm plus its config" indirection for newer models without coupling
them to the GPT2 implementation.
"""

import torch.nn as nn
from pydantic import BaseModel

from modalities.config.lookup_enum import LookupEnum
from modalities.models.components.layer_norms import (
    LayerNormConfig,
    PytorchRMSLayerNormConfig,
    RMSLayerNorm,
    RMSLayerNormConfig,
)


class NormTypes(LookupEnum):
    """
    Enum lookup class for normalization layers.

    Attributes:
        rms_norm: The deprecated in-house RMSLayerNorm class.
        layer_norm: nn.LayerNorm class.
        pytorch_rms_norm: nn.RMSNorm class.
    """

    rms_norm = RMSLayerNorm
    layer_norm = nn.LayerNorm
    pytorch_rms_norm = nn.RMSNorm


class NormWrapperConfig(BaseModel):
    """
    Configuration selecting a normalization layer type together with its own configuration.

    Attributes:
        norm_type (NormTypes): Which normalization implementation to instantiate.
        config (PytorchRMSLayerNormConfig | RMSLayerNormConfig | LayerNormConfig): The
            constructor arguments of the selected normalization implementation.
    """

    norm_type: NormTypes
    config: PytorchRMSLayerNormConfig | RMSLayerNormConfig | LayerNormConfig

    def build(self) -> nn.Module:
        """
        Instantiates a fresh normalization module.

        A new instance is created on every call so that each layer of a model owns its own
        normalization parameters.

        Returns:
            nn.Module: The instantiated normalization module.
        """
        return self.norm_type.value(**dict(self.config))
