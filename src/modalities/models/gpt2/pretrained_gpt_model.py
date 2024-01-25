from typing import Dict

import torch
from transformers import PreTrainedModel

from modalities.config.config import PretrainedGPTConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM
import torch.nn as nn


class PretrainedGPTModel(PreTrainedModel):
    config_class = PretrainedGPTConfig

    def __init__(self, config: PretrainedGPTConfig, model: nn.Module = None):
        super().__init__(config)
        # TODO offloading the parameters like this is ugly
        if model is None:
            self.model: GPT2LLM = GPT2LLM(**dict(config.config))
        else:
            self.model = model

    def forward(self, tensor):
        model_input = {"input_ids": tensor}
        model_forward_output: Dict[str, torch.Tensor] = self.model.forward(model_input)
        return model_forward_output[self.config.config.prediction_key]