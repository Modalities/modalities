from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, GetCoreSchemaHandler, validator
from pydantic_core import core_schema


class ComponentIF:
    def print(self) -> None:
        print("ComponentIF")


class ComponentZ:
    def __init__(self, val_2: str) -> None:
        self.val_2 = val_2


class ComponentY(ComponentIF):
    def __init__(self, val_2: str) -> None:
        self.val_2 = val_2


class ComponentX:
    def __init__(self, val_1: str, comp_if: ComponentIF) -> None:
        self.val_1 = val_1
        self.comp_if = comp_if

    def print(self) -> None:
        print("ComponentX")


class ComponentTypes(Enum):
    ComponentX = ComponentX


# ====================================================


class PydanticComponentIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(ComponentIF),
            python_schema=core_schema.is_instance_schema(ComponentIF),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda instance: instance.x),
        )


PydanticComponentIFType = Annotated[ComponentIF, PydanticComponentIF]
# ====================================================


class CompXConfig(BaseModel):
    type_hint: Literal[ComponentTypes.ComponentX]
    val_1: str
    comp_if: PydanticComponentIFType

    @validator("type_hint", pre=True, always=True)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            key = ComponentTypes[key]
            return key
        return key


if __name__ == "__main__":
    config_dict = {"type_hint": ComponentTypes.ComponentX, "val_1": "val 1", "comp_if": ComponentZ("val 2")}
    comp_x_config = CompXConfig(**config_dict)
    print(comp_x_config)
