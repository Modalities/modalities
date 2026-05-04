from typing import Annotated

from pydantic import BaseModel, Field

from modalities.config.pydantic_if_types import (
    PydanticDeviceMeshIFType,
    PydanticPytorchModuleOrListType,
    PydanticPytorchModuleType,
)
from modalities.training.gradient_clipping.fsdp_gradient_clipper import GradientClippingMode
from modalities.utils.deprecated_alias import add_deprecated_alias
from modalities.utils.logger_utils import get_logger

logger = get_logger("fsdp_gradient_clipper_config")


class FSDP1GradientClipperConfig(BaseModel):
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


@add_deprecated_alias("model_parts", "wrapped_model")
class FSDP2GradientClipperConfig(BaseModel):
    """
    Configuration class for FSDP gradient clipper.

    Args:
        max_norm (float): The maximum norm value for gradient clipping.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        model_parts (PydanticPytorchModuleOrListType): The wrapped PyTorch model (parts).
        device_mesh (PydanticDeviceMeshIFType | None): The device mesh configuration.

    Attributes:
        max_norm (float): The maximum norm value for gradient clipping.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        model_parts (PydanticPytorchModuleOrListType): The wrapped PyTorch model (parts).
        device_mesh (PydanticDeviceMeshIFType | None): The device mesh configuration.
    """

    max_norm: Annotated[float, Field(strict=True, gt=0)]
    norm_type: GradientClippingMode
    model_parts: PydanticPytorchModuleOrListType
    device_mesh: PydanticDeviceMeshIFType


class FSDP1DummyGradientClipperConfig(BaseModel):
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


@add_deprecated_alias("model_parts", "wrapped_model")
class FSDP2DummyGradientClipperConfig(BaseModel):
    """
    Configuration class for FSDP dummy gradient clipper.

    Args:
        model_parts (PydanticPytorchModuleOrListType): The wrapped PyTorch model (parts).
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        device_mesh (PydanticDeviceMeshIFType | None): The device mesh configuration.

    Attributes:
        model_parts (PydanticPytorchModuleOrListType): The wrapped PyTorch model (parts).
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        device_mesh (PydanticDeviceMeshIFType | None): The device mesh configuration.
    """

    model_parts: PydanticPytorchModuleOrListType
    norm_type: GradientClippingMode
    device_mesh: PydanticDeviceMeshIFType
