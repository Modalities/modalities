from typing import Annotated

from pydantic import BaseModel, Field


class LayerNormConfig(BaseModel):
    """
    Configuration class for Layer Normalization.

    Args:
        normalized_shape (int): The expected size of the input shape.
        eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-6.
        elementwise_affine (bool, optional): Whether to include learnable affine parameters. Defaults to True.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
    """

    normalized_shape: Annotated[int, Field(strict=True, ge=1)]
    eps: Annotated[float, Field(strict=True, gt=0, default=1e-6)]
    elementwise_affine: Annotated[bool, Field(strict=True, default=True)]
    bias: Annotated[bool, Field(strict=True, default=True)]


class RMSLayerNormConfig(BaseModel):
    """
    Configuration class for RMSLayerNorm.

    Args:
        ndim (int): Number of dimensions for the input tensor. Must be greater than or equal to 1.
        epsilon (float, optional): Small value added to the input to avoid division by zero. Defaults to 1e-6.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
    """

    normalized_shape: Annotated[int, Field(strict=True, ge=1)]
    eps: Annotated[float, Field(strict=True, gt=0, default=1e-5)]
