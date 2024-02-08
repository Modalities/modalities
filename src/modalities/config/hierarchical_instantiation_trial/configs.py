from enum import Enum
from typing import Literal, Optional, TypeVar

from pydantic import BaseModel, validator

from modalities.config.hierarchical_instantiation_trial.components import ComponentTypes

# ==========CONFIGS==========

CompConfigUnion = TypeVar("CompConfigUnion")


class ReferenceConfig(BaseModel):
    class PassType(Enum):
        BY_VALUE = "by_value"
        BY_REFERENCE = "by_reference"

    instance_key: str
    pass_type: PassType

    @validator("pass_type", pre=True, always=True)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            key = ReferenceConfig.PassType[key]
            return key
        return key


class CompVConfigIF(BaseModel):
    type_hint: Literal[ComponentTypes.COMP_V]
    val_0: str


class CompVConfig(BaseModel):
    type_hint: Literal[ComponentTypes.COMP_V]
    val_0: str

    @validator("type_hint", pre=True, always=True)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            key = ComponentTypes[key]
            return key
        return key


class CompWConfig(BaseModel):
    type_hint: Literal[ComponentTypes.COMP_W]
    val_0: str
    # Problem: We cannot know all the config types at this point
    # and might want to extend the config types dynamically
    comp_v_config: Optional[CompVConfig | ReferenceConfig] = None

    @validator("type_hint", pre=True, always=True)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            key = ComponentTypes[key]
            return key
        return key


class CompXConfig(BaseModel):
    type_hint: Literal[ComponentTypes.COMP_X]
    val_1: str

    @validator("type_hint", pre=True, always=True)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            key = ComponentTypes[key]
            return key
        return key


class CompYConfig(BaseModel):
    type_hint: Literal[ComponentTypes.COMP_Y]
    val_2: str
    comp_x_config: CompXConfig | ReferenceConfig
    comp_w_config: CompWConfig | ReferenceConfig

    @validator("type_hint", pre=True, always=True)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            key = ComponentTypes[key]
            return key
        return key


class CompZConfig(BaseModel):
    type_hint: Literal[ComponentTypes.COMP_Z]
    val_3: str
    comp_y_config: CompYConfig | ReferenceConfig

    @validator("type_hint", pre=True, always=True)
    def _string_to_enum(cls, key: str):
        if isinstance(key, str):
            key = ComponentTypes[key]
            return key
        return key


class ComponentConfigTypes(Enum):
    COMP_V = CompVConfig
    COMP_W = CompWConfig
    COMP_X = CompXConfig
    COMP_Y = CompYConfig
    COMP_Z = CompZConfig
    REFERENCE_CONFIG = ReferenceConfig


# ==========RESOLVER=REGISTER==========

# class ResolverRegister:
#     def __init__(self) -> None:
#         self._component_type_registry: Dict[str, Type] = {}
#         self._config_type_registry: Dict[str, Type] = {}

#     def _build_component_by_config_dict(self, config_dict: Dict) -> Any:
#         def _build_component_config(config: Dict) -> ComponentConfig:
#             component_type_hint = self._component_type_registry[config["type_hint"]]
#             config_type_hint = self._config_type_registry[config["type_hint"]]

#             config = ComponentConfig[component_type_hint, config_type_hint].model_validate(config_dict)
#             return config

#     def verify_integrity(self, app_config: Dict) -> None:
#         # enforce all instance_keys to be unique
#         # otherwise, there will be a reference mismatch when building the components
#         # check that there are no reference loops, e.g., component a references
#         # component b, which references component a
#         pass

#     @staticmethod
#     def build_components(app_config: Dict, component_names: List[str]) -> Dict[str, Any]:
#         for config in app_config:
#             ResolverRegister.build_component_by_config(config)

#     @staticmethod
#     def _base_model_to_dict_non_recursive(base_model: BaseModel) -> Dict[str, Any]:
#         return {key: getattr(base_model, key) for key in base_model.model_dump().keys()}

#     @staticmethod
#     def build_component_by_config(config: ComponentConfig) -> Any:
#         if config.pass_by_value is not None:
#             pass
#         if config.pass_by_reference is not None:
#             pass
#         else:
#             # instantiate component
#             pass


# # ==========MAIN==========
# if __name__ == "__main__":
#     config_dict = load_app_config_dict(
#         config_file_path=Path("/raid/s3/opengptx/max_lue/modalities/src/modalities/config/config.yaml")
#     )
#     comp_w_2_config = config_dict["comp_w_2_config"]
#     type_hint_string = comp_w_2_config["type_hint"]

#     comp_cls, config_cls = Components1Enum.COMP_W.value
#     config = config_cls(val_0="some value")
#     obj = comp_cls(**config.model_dump())
#     print(obj.val_0)


# config_dict = {"<config_path>": "<obj>"}
# component_dict = {"<config path>": "<obj>"}
#
# Traverse config
#    for sub_config in config_dict.sub_configs (component configs or reference configs):
#         traverse sub_config
#
#    if config_dict is "by_type" reference
#       if reference does not exist
#           traverse reference config
#   if config_dict has no sub configs in "config" field:
#       config =  instantiate(config_dict)
#       instantiate component via type hint and config_dict
#       add component to component_dict
#
