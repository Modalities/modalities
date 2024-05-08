from typing import Annotated

from pydantic import BaseModel, Field

from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.training.gradient_clipping.fsdp_gradient_clipper import GradientClippingMode


class FSDPGradientClipperConfig(BaseModel):
    max_norm: Annotated[float, Field(strict=True, gt=0)]
    norm_type: GradientClippingMode
    wrapped_model: PydanticPytorchModuleType


class FSDPDummyGradientClipperConfig(BaseModel):
    wrapped_model: PydanticPytorchModuleType
    norm_type: GradientClippingMode


class DummyGradientClipperConfig(BaseModel):
    pass
