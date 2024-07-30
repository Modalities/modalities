from enum import Enum
from typing import Dict

from pydantic import BaseModel

from modalities.component_instantiation.component_factory import ComponentFactory
from modalities.component_instantiation.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.component_instantiation.registry.components import COMPONENTS
from modalities.component_instantiation.registry.registry import Registry


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
