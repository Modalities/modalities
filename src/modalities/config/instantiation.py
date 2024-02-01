from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel

# yaml example for AppConfig
example_app_config_yaml = """
comp_w:
    type_hint: CompW
    config:
        val_0: "some value"
    instance_key: "comp_w_instance"

comp_z:
    type_hint: CompZ
    config:
        val_3: "some other value"
        comp_y_config:
            type_hint: CompY
            config:
                val_2: "yet another value"
                comp_x_config:
                    type_hint: CompX
                    config:
                        val_1: "final value"
                comp_w_config:
                    type_hint: CompW
                    reference_key: "comp_w_instance" 
"""


# ==========COMPONTENTS==========


class CompW:
    def __init__(self, val_0: str) -> None:
        self.val_0 = val_0


class CompX:
    def __init__(self, val_1: str) -> None:
        self.val_1 = val_1


class CompY:
    def __init__(self, val_2: str, comp_x: CompX) -> None:
        self.val_2 = val_2
        self.comp_x = comp_x


class CompZ:
    def __init__(self, val_3: str, comp_y: CompY) -> None:
        self.val_3 = val_3
        self.comp_y = comp_y


# ======COMPONENT=ENUMS======


class Components1Enum(Enum):
    CompW = CompW
    CompX = CompX


class Components2Enum(Enum):
    CompY = CompY
    CompZ = CompZ


# ==========COMPONENT=CONFIGS==========

T = TypeVar("T")
U = TypeVar("U")


# Wrapping Configs
class ComponentConfig(Generic[T, U], BaseModel):
    type_hint: T
    config: Optional[U] = None
    instance_key: Optional[str] = None


class PassType(Enum):
    BY_VALUE = "by_value"
    BY_REFERENCE = "by_reference"


class ReferenceConfig(BaseModel):
    instance_key: str = None
    pass_type: PassType = None


# Concrete ComponentConfigs


class CompWConfig(BaseModel):
    val_0: str


class CompXConfig(BaseModel):
    val_1: str


class CompYConfig(BaseModel):
    val_2: str
    comp_x_config: ComponentConfig[Components1Enum.CompX, CompXConfig] | ReferenceConfig
    comp_w_config: ComponentConfig[Components1Enum.CompW, CompWConfig] | ReferenceConfig


class CompZConfig(BaseModel):
    val_3: str
    comp_y_config: ComponentConfig[Components2Enum.CompY, CompYConfig] | ReferenceConfig


# Entry Config


class AppConfig(BaseModel):
    comp_w: ComponentConfig[Components1Enum.CompW, CompWConfig] | ReferenceConfig
    comp_z: ComponentConfig[Components2Enum.CompZ, CompZConfig] | ReferenceConfig


# ==========RESOLVER=REGISTER==========


class ResolverRegister:
    def __init__(self) -> None:
        # we don't need to register the types, as they are already stored in the
        # type_hint field of the ComponentConfig
        pass

    def verify_integrity(self, app_config: AppConfig) -> None:
        # enforce all instance_keys to be unique
        # otherwise, there will be a reference mismatch when building the components
        # check that there are no reference loops, e.g., component a references
        # component b, which references component a
        pass

    @staticmethod
    def build_components(app_config: AppConfig, component_names: List[str]) -> Dict[str, Any]:
        for config in app_config:
            ResolverRegister.build_component_by_config(config)

    @staticmethod
    def _base_model_to_dict_non_recursive(base_model: BaseModel) -> Dict[str, Any]:
        return {key: getattr(base_model, key) for key in base_model.model_dump().keys()}

    @staticmethod
    def build_component_by_config(config: ComponentConfig) -> Any:
        if config.pass_by_value is not None:
            pass
        if config.pass_by_reference is not None:
            pass
        else:
            # instantiate component
            pass
