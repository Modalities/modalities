from typing import Annotated, List

from pydantic import BaseModel

from modalities.config.config import PydanticThirdPartyTypeIF
from tests.config.components import Component_V_W_X_IF

# from abc import ABC
# from enum import Enum
# from pydantic import BaseModel
# from tests.config.compenents import ComponentTypes
# class CompConfigABC(BaseModel, ABC):
#     type_hint: Enum
#
#     @field_validator("type_hint", mode="before", check_fields=False)
#     def _string_to_enum(cls, key: str):
#         if isinstance(key, str):
#             try:
#                 key = ComponentTypes[key]
#             except KeyError as e:
#                 raise ValueError(f"{key} is not a valid ComponentType") from e
#             return key
#         return key

# class PassType(Enum):
#     BY_VALUE = "by_value"
#     BY_REFERENCE = "by_reference"


# class ReferenceConfig(BaseModel):
#     instance_key: str
#     pass_type: PassType

#     @validator("pass_type", pre=True)
#     def _string_to_enum(cls, key: str):
#         if isinstance(key, str):
#             try:
#                 key = PassType[key]
#             except KeyError as e:
#                 raise ValueError(f"{key} is not a valid PassType") from e
#             return key
#         return key

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
    multi_dependency: List[PydanticComponent_V_W_X_IF_Type]


class CompZConfig(BaseModel):
    val_z: str
