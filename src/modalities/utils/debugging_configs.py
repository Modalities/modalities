from pydantic import BaseModel

from modalities.config.pydantic_if_types import PydanticPytorchModuleType, PydanticRemovableHandleType


class DebuggingConfig(BaseModel):
    forward_hooks: list[list[PydanticRemovableHandleType]] = []
    """List of lists of forward hook handles registered on the model."""

    enable_determinism: bool = False
    """Whether to enable deterministic operations in PyTorch for debugging purposes."""


class NaNHookConfig(BaseModel):
    """Configuration for registering NaN detection hooks on a model."""

    model: PydanticPytorchModuleType
    raise_exception: bool = False
    """Whether to raise an exception when NaN is detected."""


class PrintForwardHookConfig(BaseModel):
    """Configuration for registering print hooks on a model."""

    model: PydanticPytorchModuleType
    print_shape_only: bool = False
