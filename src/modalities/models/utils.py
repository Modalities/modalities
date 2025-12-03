from enum import Enum

import torch.nn as nn
from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ConfigDictType
from modalities.config.pydantic_if_types import PydanticAppStateType, PydanticPytorchModuleType
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


class ModelTypeEnum(Enum):
    """
    Enumeration class representing different types of models.

    Attributes:
        MODEL (str): Represents a regular model.
        CHECKPOINTED_MODEL (str): Represents a checkpointed model.
        DCP_CHECKPOINTED_MODEL (str): Represents a distributed checkpointed model.
    """

    MODEL = "model"
    CHECKPOINTED_MODEL = "checkpointed_model"
    DCP_CHECKPOINTED_MODEL = "dcp_checkpointed_model"


def get_model_from_config(config: ConfigDictType, model_type: ModelTypeEnum) -> nn.Module:
    """
    Retrieves a model from the given configuration based on the specified model type.

    Args:
        config (ConfigDictType): The configuration dictionary.
        model_type (ModelTypeEnum): The type of the model to retrieve.

    Returns:
        nn.Module: The model object based on the specified model type.

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

    elif model_type.value == "dcp_checkpointed_model":

        class PydanticConfig(BaseModel):
            app_state: PydanticAppStateType

            @property
            def dcp_checkpointed_model(self) -> PydanticPytorchModuleType:
                return self.app_state.model

    else:
        raise NotImplementedError()

    components = component_factory.build_components(config_dict=config, components_model_type=PydanticConfig)
    return getattr(components, model_type.value)
