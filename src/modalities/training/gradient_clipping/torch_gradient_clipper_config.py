from typing import Annotated

from pydantic import BaseModel, Field

from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.training.gradient_clipping.fsdp_gradient_clipper import GradientClippingMode


class TorchGradientClipperConfig(BaseModel):
    """
    Configuration class for torch gradient clipper.

    Args:
        max_norm (float): The maximum norm value for gradient clipping.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        model (PydanticPytorchModuleType): The PyTorch model.

    Attributes:
        max_norm (float): The maximum norm value for gradient clipping.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        model (PydanticPytorchModuleType): The PyTorch model.
    """

    max_norm: Annotated[float, Field(strict=True, gt=0)]
    norm_type: GradientClippingMode
    model: PydanticPytorchModuleType


class TorchDummyGradientClipperConfig(BaseModel):
    """
    Configuration class for torch dummy gradient clipper.

    Args:
        model (PydanticPytorchModuleType): The PyTorch model.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.

    Attributes:
        model (PydanticPytorchModuleType): The PyTorch model.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
    """

    model: PydanticPytorchModuleType
    norm_type: GradientClippingMode
