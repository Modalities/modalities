from abc import ABC
from enum import Enum
from typing import Annotated, Any, List, Literal

from pydantic import BaseModel, GetCoreSchemaHandler, validator
from pydantic_core import core_schema

from tests.config.components import Component_V_W_X_IF, ComponentTypes, CustomComponentTypes


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

    @validator("type_hint", pre=True, allow_reuse=True, check_fields=False)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            try:
                key = ComponentTypes[key]
            except KeyError as e:
                raise ValueError(f"{key} is not a valid ComponentType") from e
            return key
        return key


class PassType(Enum):
    BY_VALUE = "by_value"
    BY_REFERENCE = "by_reference"


class ReferenceConfig(BaseModel):
    instance_key: str
    pass_type: PassType

    @validator("pass_type", pre=True)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            try:
                key = PassType[key]
            except KeyError as e:
                raise ValueError(f"{key} is not a valid PassType") from e
            return key
        return key


class CompVConfig(CompConfigABC):
    type_hint: Literal[ComponentTypes.COMP_V]
    val_v: str


class CompWConfig(CompConfigABC):
    type_hint: Literal[ComponentTypes.COMP_W]
    val_w: str


class CompXConfig(CompConfigABC):
    type_hint: Literal[ComponentTypes.COMP_X]
    val_x: str
    single_dependency: PydanticComponen_V_W_X_IF_Type


class CompYConfig(CompConfigABC):
    type_hint: Literal[ComponentTypes.COMP_Y]
    val_y: str
    multi_dependency: List[PydanticComponen_V_W_X_IF_Type]


class CompZConfig(CompConfigABC):
    type_hint: Literal[ComponentTypes.COMP_Z]
    val_z: str


class CustomCompConfigABC(BaseModel, ABC):
    # TODO make this a string and then implement the mapping
    # to the class outside of the basemodel (i.e. in the factory)
    type_hint: Enum

    @validator("type_hint", pre=True, allow_reuse=True, check_fields=False)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            try:
                key = CustomComponentTypes[key]
            except KeyError as e:
                raise ValueError(f"{key} is not a valid ComponentType") from e
            return key
        return key


class CustomComp1Config(CustomCompConfigABC):
    type_hint: Literal[CustomComponentTypes.CUSTOM_COMP_1]
    val_1: str
