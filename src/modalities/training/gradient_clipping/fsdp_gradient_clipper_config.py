from typing import Annotated

from pydantic import BaseModel, Field

from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.training.gradient_clipping.fsdp_gradient_clipper import GradientClippingMode


class FSDPGradientClipperConfig(BaseModel):
    """
    Configuration class for FSDP gradient clipper.

    Args:
        max_norm (float): The maximum norm value for gradient clipping.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        wrapped_model (PydanticPytorchModuleType): The wrapped PyTorch model.

    Attributes:
        max_norm (float): The maximum norm value for gradient clipping.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        wrapped_model (PydanticPytorchModuleType): The wrapped PyTorch model.
    """

    max_norm: Annotated[float, Field(strict=True, gt=0)]
    norm_type: GradientClippingMode
    wrapped_model: PydanticPytorchModuleType


class FSDPDummyGradientClipperConfig(BaseModel):
    """
    Configuration class for FSDP dummy gradient clipper.

    Args:
        wrapped_model (PydanticPytorchModuleType): The wrapped PyTorch model.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.

    Attributes:
        wrapped_model (PydanticPytorchModuleType): The wrapped PyTorch model.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
    """

    wrapped_model: PydanticPytorchModuleType
    norm_type: GradientClippingMode


class DummyGradientClipperConfig(BaseModel):
    """
    Configuration class for dummy gradient clipper.

    This class is a placeholder and does not have any specific functionality.

    Attributes:
        None

    Methods:
        None
    """

    pass
