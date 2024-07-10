from typing import Annotated

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from modalities.config.lookup_enum import LookupEnum


class RMSLayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5):
        """
        Initialize the RMSNorm normalization layer.
        Original paper: https://arxiv.org/pdf/1910.07467.pdf
        Source code adopted from https://github.com/facebookresearch/llama/blob/a0a4da8b497c566403941ceec47c2512ecf9dd20/llama/model.py#L34C1-L77C36

        Args:
            ndim (int): The dimension of the input tensor.
            epsilon (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
            bias (bool, optional): If True, the layer will learn an additive bias. Default is True.
        """
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(ndim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.bias = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.bias is None:
            return output * self.weight
        else:
            return output * self.weight + self.bias


class LayerNorms(LookupEnum):
    """
    An enumeration of the different layer normalization techniques.
    """

    RMSNorm = RMSLayerNorm
    LayerNorm = nn.LayerNorm


class LayerNormConfig(BaseModel):
    normalized_shape: Annotated[int, Field(strict=True, ge=1)]
    eps: Annotated[float, Field(strict=True, gt=0, default=1e-6)]
    elementwise_affine: Annotated[bool, Field(strict=True, default=True)]
    bias: Annotated[bool, Field(strict=True, default=True)]


class RMSLayerNormConfig(BaseModel):
    ndim: Annotated[int, Field(strict=True, ge=1)]
    epsilon: Annotated[float, Field(gt=0, default=1e-6)]
    bias: Annotated[bool, Field(strict=True, default=True)]
