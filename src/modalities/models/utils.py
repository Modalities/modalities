from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


def get_model_from_config(config: dict):
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    class ModelConfig(BaseModel):
        model: PydanticPytorchModuleType

    components = component_factory.build_components(config_dict=config, components_model_type=ModelConfig)
    return components.model
