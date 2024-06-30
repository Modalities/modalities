from enum import Enum
from typing import Annotated, List, Optional

import torch.nn as nn
from pydantic import BaseModel, Field, root_validator


class ActivationType(str, Enum):
    GELU = "gelu"
    FUSED_SWIGLU = "fused_swiglu"


class ModuleTypes(Enum):
    linear: nn.Linear
    embedding: nn.Embedding


class ModuleTypeFilter(BaseModel):
    module_type: ModuleTypes  # here we filter for the type of the model, e.g., nn.Linear
    apply_to_bias: bool
    apply_to_weights: bool


class PlainWeightInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    hidden_dim: Optional[int] = None

    @root_validator
    def check_std_and_hidden_dim(cls, values):
        std = values.get("std")
        hidden_dim = values.get("hidden_dim")
        if std == "auto" and hidden_dim is None:
            raise ValueError("hidden_dim must be specified when std is 'auto'")
        return values


class ScaledWeightInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    parameter_name_suffixes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"
