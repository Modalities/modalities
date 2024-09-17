from enum import Enum
from typing import Dict

from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


class ModelTypeEnum(Enum):
    """
    Enumeration class representing different types of models.

    Attributes:
        MODEL (str): Represents a regular model.
        CHECKPOINTED_MODEL (str): Represents a checkpointed model.
    """

    MODEL = "model"
    CHECKPOINTED_MODEL = "checkpointed_model"


def get_model_from_config(config: Dict, model_type: ModelTypeEnum):
    """
    Retrieves a model from the given configuration based on the specified model type.

    Args:
        config (Dict): The configuration dictionary.
        model_type (ModelTypeEnum): The type of the model to retrieve.

    Returns:
        Any: The model object based on the specified model type.

    Raises:
        NotImplementedError: If the model type is not supported.
    """
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
