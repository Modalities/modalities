from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizer

from modalities.config.lookup_enum import LookupEnum
from modalities.models.model import NNModel


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


class HuggingFaceModelTypes(LookupEnum):
    AutoModelForCausalLM = AutoModelForCausalLM
    AutoModelForMaskedLM = AutoModelForMaskedLM


class HuggingFacePretrainedModelConfig(BaseModel):
    model_type: HuggingFaceModelTypes
    model_name: Path
    prediction_key: str
    huggingface_prediction_subscription_key: str
    sample_key: str
    model_args: Optional[Any] = None
    kwargs: Optional[Any] = None


class HuggingFacePretrainedModel(NNModel):

    def __init__(
            self,
            model_type: HuggingFaceModelTypes,
            model_name: str,
            prediction_key: str,
            huggingface_prediction_subscription_key: str,
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
        self.huggingface_prediction_subscription_key = huggingface_prediction_subscription_key
        self.sample_key = sample_key

        # NOTE: If the model needs to be downloaded, it is NOT necessary to guard the access for rank 0.
        # This is taken care of internally in huggingface hub see:
        # https://github.com/huggingface/huggingface_hub/blob/3788f537b10c7d02149d6bf017d2ce19885f90a2/src/huggingface_hub/file_download.py#L1457
        self.huggingface_model = model_type.value.from_pretrained(
            model_name, local_files_only=False, *model_args, **kwargs
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.huggingface_model.forward(inputs[self.sample_key])
        return {self.prediction_key: output[self.huggingface_prediction_subscription_key]}

    @property
    def fsdp_block_names(self) -> List[str]:
        return self.huggingface_model._no_split_modules


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
    model = HuggingFacePretrainedModel(
        model_type=HuggingFaceModelTypes.AutoModelForCausalLM,
        model_name="epfl-llm/meditron-7b",
        prediction_key="logits",
        huggingface_prediction_subscription_key="logits",
        sample_key="input_ids",
    )
    print(model)
