import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from modalities.config.config import CheckpointedModelConfig
from modalities.checkpointing.torch.torch_checkpoint_loading import TorchCheckpointLoading
from modalities.models.model_factory import ModelFactory


class HuggingFaceAdapterConfig(ABC, PretrainedConfig):
    model_type = "modalities"

    def __init__(self, config_dict = None, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = config_dict
        # TODO Iterate over all dict and change PosixPath to str
        if config_dict:
            self.config_dict["settings"]["config_file_path"] = str(self.config_dict["settings"]["config_file_path"])

    # # TODO check if this is still needed
    # def to_json_string(self, use_diff: bool = True) -> str:
    #     cfg = dict(config=self.config_dict.model_dump(), model_type=self.model_type)
    #     return json.dumps(cfg)


class HuggingFaceModel(PreTrainedModel):
    config_class = HuggingFaceAdapterConfig

    def __init__(self, config: HuggingFaceAdapterConfig, model: Optional[nn.Module] = None):
        super().__init__(config)
        # TODO pass correct model type to __init__
        self.model = model

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
