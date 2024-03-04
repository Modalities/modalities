from typing import Annotated

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.nn import functional as F

from modalities.config.lookup_enum import LookupEnum


class LayerNormIF(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RMSLayerNorm(LayerNormIF):
    def __init__(self, ndim: int, epsilon: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.
        Original paper: https://arxiv.org/pdf/1910.07467.pdf
        Source code adopted from https://github.com/facebookresearch/llama/blob/a0a4da8b497c566403941ceec47c2512ecf9dd20/llama/model.py#L34C1-L77C36

        Args:
            ndim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(ndim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ZLayerNorm(LayerNormIF):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input=x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.epsilon,
        )


class LayerNorms(LookupEnum):
    """
    An enumeration of the different layer normalization techniques.
    """

    RMSNorm = RMSLayerNorm
    ZLayerNorm = ZLayerNorm


class ZLayerNormConfig(BaseModel):
    ndim: Annotated[int, Field(strict=True, ge=1)]
    bias: Annotated[bool, Field(default=True)]
    epsilon: Annotated[float, Field(gt=0, default=1e-6)]


class RMSLayerNormConfig(BaseModel):
    ndim: Annotated[int, Field(strict=True, ge=1)]
    epsilon: Annotated[float, Field(gt=0, default=1e-6)]
