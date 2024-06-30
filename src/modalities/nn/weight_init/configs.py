from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, model_validator

from modalities.config.pydanctic_if_types import PydanticWeightInitializationIFType


class WeightInitializerWrapperConfig(BaseModel):
    weight_initializers: List[PydanticWeightInitializationIFType]


class PlainWeightInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    hidden_dim: Optional[int] = None

    @model_validator(mode="after")
    def check_std_and_hidden_dim(self):
        if self.std == "auto" and self.hidden_dim is None:
            raise ValueError("hidden_dim must be specified when std is 'auto'")
        return self


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
