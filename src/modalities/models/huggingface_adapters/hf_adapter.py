import json
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from modalities.exceptions import ConfigError
from modalities.models.model import NNModel
from modalities.models.utils import get_model_from_config, ModelTypeEnum


class HFModelAdapterConfig(PretrainedConfig):
    model_type = "modalities"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.config is added by the super class via kwargs
        if self.config is None:
            raise ConfigError("Config is not passed in HFModelAdapterConfig")
        # since the config will be saved to json and json can't handle posixpaths, we need to convert them to strings
        self._convert_posixpath_to_str(self.config)

    def to_json_string(self, use_diff: bool = True) -> str:
        if self.config:
            json_dict = {"config": self.config.copy(), "model_type": self.model_type}
        else:
            json_dict = {}
        return json.dumps(json_dict)

    def _convert_posixpath_to_str(self, d: dict):
        """
        Recursively iterate over the dictionary and convert PosixPath values to strings.
        """
        for key, value in d.items():
            if isinstance(value, PosixPath):
                d[key] = str(value)
            elif isinstance(value, dict):
                self._convert_posixpath_to_str(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, PosixPath):
                        d[key] = str(item)
                    elif isinstance(item, dict):
                        self._convert_posixpath_to_str(item)
                    else:
                        d[key] = item


class HFModelAdapter(PreTrainedModel):
    config_class = HFModelAdapterConfig

    def __init__(self, config: HFModelAdapterConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model: NNModel = get_model_from_config(config.config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
        assert hasattr(self.model, "prediction_key"), "Missing entry model.prediction_key in config"

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
            return model_forward_output[self.model.prediction_key]
        else:
            return ModalitiesModelOutput(**model_forward_output)


    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None, **kwargs
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
