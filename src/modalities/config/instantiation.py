from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from omegaconf import OmegaConf
from pydantic import BaseModel, validator

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


# ==========COMPONENT=CONFIGS==========

T = TypeVar("T")
U = TypeVar("U")


# Wrapping Configs
class ComponentConfig(BaseModel, Generic[T, U]):
    type_hint: Type[T]
    config: Optional[U] = None

    # @validator('type_hint', pre=True, always=True)
    # def validate_type_hint(cls, v):
    #     if not isinstance(v, type):
    #         raise ValueError("type_hint must be a type")
    #     return v


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
    comp_x_config: ComponentConfig[Components1Enum, CompXConfig] | ReferenceConfig
    comp_w_config: ComponentConfig[Components1Enum.COMP_W, CompWConfig] | ReferenceConfig


class CompZConfig(BaseModel):
    val_3: str
    comp_y_config: ComponentConfig[Components2Enum.COMP_Y, CompYConfig] | ReferenceConfig


# ======COMPONENT=ENUMS======


class Components1ConfigEnum(Enum):
    COMP_W = CompWConfig
    COMP_X = CompXConfig


class Components1Enum(Enum):
    # COMP_W = CompW, CompWConfig
    COMP_X = (CompX, CompXConfig)


class Components2ConfigEnum(Enum):
    COMP_Y = CompYConfig
    COMP_Z = CompZConfig


class Components2Enum(Enum):
    COMP_Y = (CompY, CompYConfig)
    COMP_Z = (CompZ, CompZConfig)


# ==========RESOLVER=REGISTER==========


class ResolverRegister:
    def __init__(self) -> None:
        self._component_type_registry: Dict[str, Type] = {}
        self._config_type_registry: Dict[str, Type] = {}

    def _build_component_by_config_dict(self, config_dict: Dict) -> Any:
        def _build_component_config(config: Dict) -> ComponentConfig:
            component_type_hint = self._component_type_registry[config["type_hint"]]
            config_type_hint = self._config_type_registry[config["type_hint"]]

            config = ComponentConfig[component_type_hint, config_type_hint].model_validate(config_dict)
            return config

    def verify_integrity(self, app_config: Dict) -> None:
        # enforce all instance_keys to be unique
        # otherwise, there will be a reference mismatch when building the components
        # check that there are no reference loops, e.g., component a references
        # component b, which references component a
        pass

    @staticmethod
    def build_components(app_config: Dict, component_names: List[str]) -> Dict[str, Any]:
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


class AppConfig(BaseModel):
    __root__: Dict  #  Dict[str, Union[Cat, Dog, Bird]]

    # Custom validator to check each item in the dictionary
    @validator("__root__", pre=True, each_item=True)
    def validate_pets(cls, v, values, **kwargs):
        if isinstance(v, dict):
            # Attempt to parse the dictionary into one of the pet models
            for PetModel in []:  # (Cat, Dog, Bird):
                try:
                    return PetModel(**v)
                except ValueError:
                    continue
            raise ValueError(f"Value does not match any pet model: {v}")
        return v

    # To interact with the model as a dictionary
    def __getitem__(self, item):
        return self.__root__[item]

    def __setitem__(self, key, value):
        self.__root__[key] = value

    def __iter__(self):
        return iter(self.__root__)

    def keys(self):
        return self.__root__.keys()

    def values(self):
        return self.__root__.values()

    def items(self):
        return self.__root__.items()


def load_app_config_dict(config_file_path: Path) -> Dict:
    cfg = OmegaConf.load(config_file_path)
    return OmegaConf.to_container(cfg, resolve=True)


# ==========MAIN==========
if __name__ == "__main__":
    config_dict = load_app_config_dict(
        config_file_path=Path("/raid/s3/opengptx/max_lue/modalities/src/modalities/config/config.yaml")
    )
    comp_w_2_config = config_dict["comp_w_2_config"]
    type_hint_string = comp_w_2_config["type_hint"]

    comp_cls, config_cls = Components1Enum.COMP_W.value
    config = config_cls(val_0="some value")
    obj = comp_cls(**config.model_dump())
    print(obj.val_0)
