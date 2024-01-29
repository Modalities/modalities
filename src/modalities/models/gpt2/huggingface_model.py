from typing import Dict

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from modalities.config.config import HuggingFaceModelConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM


class HuggingFaceModel(PreTrainedModel):
    config_class = HuggingFaceModelConfig

    def __init__(self, config: HuggingFaceModelConfig, model: nn.Module = None):
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
