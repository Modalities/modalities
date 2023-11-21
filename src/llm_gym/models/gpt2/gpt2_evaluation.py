
import json
from transformers import PretrainedConfig
from llm_gym.models.gpt2.gpt2_model import GPTConfig, GPT2LLM
from transformers import PreTrainedModel
from typing import Dict
import torch


class PretrainedGPTConfig(PretrainedConfig):
    model_type = "llm_gym_gpt2"

    def __init__(
            self,
            config: GPTConfig = None,
            **kwargs
    ):
        if type(config) == dict:
            config = GPTConfig(**config)
        self.config = config

        super().__init__(**kwargs)

    def to_json_string(self, use_diff: bool = True) -> str:
        if self.config:
            json_dict = {"config": self.config.__dict__.copy()}
            json_dict["config"]["attention"] = {
                "attention_type": self.config.attention.attention_type.value,
                "scaling_factor": self.config.attention.scaling_factor
            }
            json_dict["config"]["weight_init"] = {
                "mean": self.config.weight_init.mean,
                "std": self.config.weight_init.std
            }
            json_dict["model_type"] = self.model_type
        else:
            json_dict = {}
        return json.dumps(json_dict)


class PretrainedGPTModel(PreTrainedModel):
    config_class = PretrainedGPTConfig

    def __init__(self,
                 config: PretrainedGPTConfig
                 ):
        super().__init__(config)
        self.model: GPT2LLM = GPT2LLM(config=config.config)

    def forward(self, tensor):
        model_input = {"input_ids": tensor}
        model_forward_output: Dict[str, torch.Tensor] = self.model.forward(model_input)
        return model_forward_output[self.config.config.prediction_key]


if __name__ == '__main__':
    ...
