from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, validator
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from modalities.models.model import NNModel
from modalities.util import parse_enum_by_name

# Huggingface Model dependencies
#
# ModuleUtilsMixin
# GenerationMixin
# PushToHubMixin
# PeftAdapterMixin
#   <- PreTrainedModel
#       <- LlamaPreTrainedModel    The bare LLaMA Model outputting raw hidden-states without any specific head on top.
#           <- LlamaModel      The bare LLaMA Model outputting raw hidden-states without any specific head on top.
#           <- LlamaForCausalLM
#           <- LlamaForSequenceClassification    The LLaMa transformer with a sequence classif. head on top (lin. layer)


class HuggingFaceModelTypes(Enum):
    AutoModelForCausalLM = AutoModelForCausalLM
    AutoModelForMaskedLM = AutoModelForMaskedLM


class HuggingFacePretrainedModelConfig(BaseModel):
    model_type: HuggingFaceModelTypes
    model_name: Path
    prediction_key: str
    sample_key: str
    model_args: Optional[Any] = None
    kwargs: Optional[Any] = None

    @validator("model_type", pre=True, always=True)
    def parse_sharding_strategy_by_name(cls, name):
        return parse_enum_by_name(name=name, enum_type=HuggingFaceModelTypes)


class HuggingFacePretrainedModel(NNModel):
    def __init__(
        self,
        model_type: HuggingFaceModelTypes,
        model_name: str,
        prediction_key: str,
        sample_key: str,
        model_args: Optional[Any] = None,
        kwargs: Optional[Any] = None,
    ):
        super().__init__()
        if model_args is None:
            model_args = []
        if kwargs is None:
            kwargs = {}
        self.prediction_key = prediction_key
        self.sample_key = sample_key

        # TODO this would be perfect for a factory design, however the resovler register currently does not
        # support functions instead of classes within enums.
        self.huggingface_model = model_type.value.from_pretrained(model_name, *model_args, **kwargs)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.huggingface_model.forward(inputs[self.sample_key])
        return {self.prediction_key: output.logits}

    @property
    def fsdp_block_names(self) -> List[str]:
        return self.huggingface_model._no_split_modules


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
    model = HuggingFacePretrainedModel(
        model_type=HuggingFaceModelTypes.AutoModelForCausalLM,
        model_name="epfl-llm/meditron-7b",
        prediction_key="logits",
    )
    print(model)
