import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from modalities.models.mamba.mamba_config import MambaBlockConfig, MixerModelConfig
from modalities.models.mamba.mamba_model import MambaLLM


class HuggingFaceAdapterConfig(ABC, PretrainedConfig):
    def __init__(self, config_dict=None, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = config_dict
        if self.config_dict:
            self.convert_posixpath_to_str(self.config_dict)

    def to_json_string(self, use_diff: bool = True) -> str:
        if self.config_dict:
            json_dict = {"config": self.config_dict.copy(), "model_type": self.model_type}
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


class HuggingFaceModelAdapter(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

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


class HuggingFaceModelAdapterConfig(ABC, BaseModel):
    ...


@dataclass
class ModalitiesModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
