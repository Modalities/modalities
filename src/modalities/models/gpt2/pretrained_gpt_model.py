from typing import Dict

import torch
from transformers import PreTrainedModel

from modalities.config.config import PretrainedGPTConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM


class PretrainedGPTModel(PreTrainedModel):
    """Pretrained GPT model class."""

    config_class = PretrainedGPTConfig

    def __init__(self, config: PretrainedGPTConfig):
        """
        Initializes a PretrainedGPTModel object.

        Args:
            config (PretrainedGPTConfig): The configuration object for the model.

        Returns:
            None
        """
        super().__init__(config)
        # TODO offloading the parameters like this is ugly
        self.model: GPT2LLM = GPT2LLM(**dict(config.config))

    def forward(self, tensor):
        """
        Forward pass of the pretrained GPT model.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        model_input = {"input_ids": tensor}
        model_forward_output: Dict[str, torch.Tensor] = self.model.forward(model_input)
        return model_forward_output[self.config.config.prediction_key]


if __name__ == "__main__":
    ...
