from transformers import PretrainedConfig
from src.llm_gym.models.gpt2.gpt2_model import GPTConfig, GPT2LLM
from transformers import PreTrainedModel
from typing import Dict
import torch


class PretrainedGPTConfig(PretrainedConfig):
    model_type = "gpt2"

    def __init__(
            self,
            config: GPTConfig,
            **kwargs
    ):
        self.config = config
        super().__init__(**kwargs)


class PretrainedGPTModel(PreTrainedModel):
    config_class = PretrainedGPTConfig

    def __init__(self,
                 config: PretrainedGPTConfig,
                 prediction_publication_key: str,  # todo get this from AppConfig.model...
                 ):
        super().__init__(config)
        self.prediction_publication_key = prediction_publication_key
        self.model: GPT2LLM = GPT2LLM(
            prediction_publication_key=self.prediction_publication_key,
            config=config.config,
        )

    def forward(self, tensor):
        model_input = {"input_ids": tensor}
        model_forward_output: Dict[str, torch.Tensor] = self.model.forward(model_input)
        return model_forward_output[self.prediction_publication_key]


if __name__ == '__main__':
    ...
