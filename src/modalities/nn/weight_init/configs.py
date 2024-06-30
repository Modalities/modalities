from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, root_validator

from modalities.nn.weight_init.weight_init_if import WeightInitializationIF


class WeightInitializerWrapperConfig(BaseModel):
    weight_initializers: List[WeightInitializationIF]


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


class NamedParameterwiseNormalInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    parameter_name_suffixes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class ScaledWeightInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    plain_std: Annotated[float, Field(strict=True, ge=0.0)]
    number_of_layers: Annotated[int, Field(strict=True, gt=0)]
    parameter_name_suffixes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class ScaledEmbedInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    parameter_name_suffixes: List[str]  # here we filter for the parameter names, e.g., "c_proj.weight"
