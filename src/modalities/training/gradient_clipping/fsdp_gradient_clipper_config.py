import warnings
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

from modalities.config.pydantic_if_types import (
    PydanticDeviceMeshIFType,
    PydanticPytorchModuleOrListType,
    PydanticPytorchModuleType,
)
from modalities.training.gradient_clipping.fsdp_gradient_clipper import GradientClippingMode
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


class FSDP2GradientClipperConfig(BaseModel):
    """
    Configuration class for FSDP gradient clipper.

    Args:
        max_norm (float): The maximum norm value for gradient clipping.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        wrapped_model (PydanticPytorchModuleType): The wrapped PyTorch model.
        device_mesh (PydanticDeviceMeshIFType | None): The device mesh configuration.

    Attributes:
        max_norm (float): The maximum norm value for gradient clipping.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        wrapped_model (PydanticPytorchModuleType): The wrapped PyTorch model.
        device_mesh (PydanticDeviceMeshIFType | None): The device mesh configuration.
    """

    max_norm: Annotated[float, Field(strict=True, gt=0)]
    norm_type: GradientClippingMode
    wrapped_model_or_parts: PydanticPytorchModuleOrListType = Field(alias="wrapped_model")
    device_mesh: PydanticDeviceMeshIFType

    @model_validator(mode="before")
    @classmethod
    def warn_deprecated_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "wrapped_model" in data:
            warnings.warn(
                "Field 'wrapped_model' is deprecated. Use 'wrapped_model_or_parts' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return data


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


class FSDP2DummyGradientClipperConfig(BaseModel):
    """
    Configuration class for FSDP dummy gradient clipper.

    Args:
        wrapped_model (PydanticPytorchModuleType): The wrapped PyTorch model.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        device_mesh (PydanticDeviceMeshIFType | None): The device mesh configuration.

    Attributes:
        wrapped_model (PydanticPytorchModuleType): The wrapped PyTorch model.
        norm_type (GradientClippingMode): The type of gradient clipping to be applied.
        device_mesh (PydanticDeviceMeshIFType | None): The device mesh configuration.
    """

    wrapped_model_or_parts: PydanticPytorchModuleOrListType = Field(alias="wrapped_model")
    norm_type: GradientClippingMode
    device_mesh: PydanticDeviceMeshIFType

    @model_validator(mode="before")
    @classmethod
    def warn_deprecated_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "wrapped_model" in data:
            warnings.warn(
                "Field 'wrapped_model' is deprecated. Use 'wrapped_model_or_parts' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return data
