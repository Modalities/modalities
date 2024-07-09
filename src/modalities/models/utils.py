from typing import Dict

from pydantic import BaseModel
from enum import Enum

from modalities.config.component_factory import ComponentFactory
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry

class ModelTypeEnum(Enum):
    MODEL = "model"
    CHECKPOINTED_MODEL = "checkpointed_model"
    
def get_model_from_config(config: Dict, model_type: ModelTypeEnum):
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    # create the pydantic config for the component factory dynamically based on model_type
    if model_type.value == "model":
        class PydanticConfig(BaseModel):
            model: PydanticPytorchModuleType
    elif model_type.value == "checkpointed_model":
        class PydanticConfig(BaseModel):
            checkpointed_model: PydanticPytorchModuleType
    else:
        raise NotImplementedError()

    components = component_factory.build_components(config_dict=config, components_model_type=PydanticConfig)
    return getattr(components, model_type.value)
