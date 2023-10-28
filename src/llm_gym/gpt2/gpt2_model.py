import torch
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Dict
import torch.nn as nn
from abc import abstractmethod


class NNModel(nn.Module):
    def __init__(self, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        super(NNModel, self).__init__()

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_impl(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in self.named_parameters()}


class GPT2LLM(NNModel):

    def __init__(self, prediction_publication_key: str, gpt_version: str = "gpt2"):
        super().__init__()
        self.prediction_publication_key = prediction_publication_key
        config = GPT2Config.from_pretrained(gpt_version, output_hidden_stages=False)
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(gpt_version, config=config)

    def forward_impl(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        output_dict = {self.prediction_publication_key: outputs.logits}
        return output_dict

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward_impl(inputs)

