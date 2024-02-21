from abc import ABC
from enum import Enum
from typing import Annotated, Any, List

from pydantic import BaseModel, GetCoreSchemaHandler, field_validator
from pydantic_core import core_schema

from tests.config.components import Component_V_W_X_IF, ComponentTypes


class PydanticComponentIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(Component_V_W_X_IF),
            python_schema=core_schema.is_instance_schema(Component_V_W_X_IF),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


PydanticComponen_V_W_X_IF_Type = Annotated[Component_V_W_X_IF, PydanticComponentIF]


class CompConfigABC(BaseModel, ABC):
    type_hint: Enum

    @field_validator("type_hint", mode="before", check_fields=False)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            try:
                key = ComponentTypes[key]
            except KeyError as e:
                raise ValueError(f"{key} is not a valid ComponentType") from e
            return key
        return key


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


class CompVConfig(BaseModel):
    val_v: str


class CompWConfig(BaseModel):
    val_w: str


class CompXConfig(BaseModel):
    val_x: str
    single_dependency: PydanticComponen_V_W_X_IF_Type


class CompYConfig(BaseModel):
    val_y: str
    multi_dependency: List[PydanticComponen_V_W_X_IF_Type]


class CompZConfig(BaseModel):
    val_z: str
