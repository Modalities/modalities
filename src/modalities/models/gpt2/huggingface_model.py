from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2HuggingFaceAdapterConfig


class HuggingFaceModel(PreTrainedModel):
    config_class = GPT2HuggingFaceAdapterConfig

    def __init__(self, config: GPT2HuggingFaceAdapterConfig, model: nn.Module = None):
        super().__init__(config)
        # TODO offloading the parameters like this is ugly
        if model is None:
            self.model: GPT2LLM = GPT2LLM(**dict(config.config))
        else:
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
            return model_forward_output[self.config.config.prediction_key]

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
