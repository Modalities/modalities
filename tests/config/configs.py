from typing import Annotated

from pydantic import BaseModel

from modalities.config.pydantic_if_types import PydanticThirdPartyTypeIF
from tests.config.components import Component_V_W_X_IF

PydanticComponent_V_W_X_IF_Type = Annotated[Component_V_W_X_IF, PydanticThirdPartyTypeIF(Component_V_W_X_IF)]


class CompVConfig(BaseModel):
    val_v: str


class CompWConfig(BaseModel):
    val_w: str


class CompXConfig(BaseModel):
    val_x: str
    single_dependency: PydanticComponent_V_W_X_IF_Type


class CompYConfig(BaseModel):
    val_y: str
    multi_dependency: list[PydanticComponent_V_W_X_IF_Type]


class CompZConfig(BaseModel):
    val_z: str
