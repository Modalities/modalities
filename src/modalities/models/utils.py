from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
    
    
def get_model_from_config(config: dict, model_type: str):
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    # create the pydantic config for the component factory dynamically based on model_type
    if model_type == "model":
        class PydanticConfig(BaseModel):
            model: PydanticPytorchModuleType
    elif model_type == "checkpointed_model":
        class PydanticConfig(BaseModel):
            checkpointed_model: PydanticPytorchModuleType
    else:
        raise NotImplementedError()
    
    components = component_factory.build_components(config_dict=config, components_model_type=PydanticConfig)
    return getattr(components, model_type)
