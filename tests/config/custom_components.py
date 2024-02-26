from abc import ABC
from enum import Enum
from typing import Literal

from pydantic import BaseModel, validator


class CustomComponent1:
    def __init__(self, val_1: str) -> None:
        self.val_1 = val_1


class CustomComponentTypes(Enum):
    CUSTOM_COMP_1 = CustomComponent1


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
