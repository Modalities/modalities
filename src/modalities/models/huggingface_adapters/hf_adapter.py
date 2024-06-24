import json
from abc import ABC
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from modalities.config.component_factory import ComponentFactory
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.models.model import NNModel
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


class HFAdapterConfig(PretrainedConfig):
    model_type = "modalities"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config:
            self.convert_posixpath_to_str(self.config)

    def to_json_string(self, use_diff: bool = True) -> str:
        if self.config:
            json_dict = {"config": self.config.copy(), "model_type": self.model_type}
        else:
            json_dict = {}
        return json.dumps(json_dict)

    def convert_posixpath_to_str(self, d):
        """
        Recursively iterate over the dictionary and convert PosixPath values to strings.
        """
        for key, value in d.items():
            if isinstance(value, PosixPath):
                d[key] = str(value)
            elif isinstance(value, dict):
                self.convert_posixpath_to_str(value)
            elif isinstance(value, list):
                d[key] = [str(item) if isinstance(item, PosixPath) else item for item in value]


class HFAdapter(PreTrainedModel):
    config_class = HFAdapterConfig

    def __init__(self, config: HFAdapterConfig, model: Optional[nn.Module] = None, *inputs, **kwargs):
        super().__init__(config)
        if not model:
            self.model: NNModel = self.get_model_from_config(config.config)
        else:
            self.model = model

    def get_model_from_config(self, config: dict):
        registry = Registry(COMPONENTS)
        component_factory = ComponentFactory(registry=registry)

        class ModelConfig(BaseModel):
            model: PydanticPytorchModuleType

        components = component_factory.build_components(
            config_dict=config, components_model_type=ModelConfig
        )
        return components.model

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
    ):
        if output_attentions or output_hidden_states:
            raise NotImplementedError
        model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        model_forward_output: Dict[str, torch.Tensor] = self.model.forward(model_input)
        if return_dict:
            return ModalitiesModelOutput(**model_forward_output)
        else:
            return model_forward_output[self.model.prediction_key]

    def prepare_inputs_for_generation(
            self, input_ids: torch.LongTensor, attention_mask=None, **kwargs
    ) -> Dict[str, Any]:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

@dataclass
class ModalitiesModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
