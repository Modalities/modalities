from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

from modalities.config.lookup_types import LookupEnum
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


class HuggingFaceModelConfig(BaseModel):
    model_type: HuggingFaceModelTypes
    pretrained_model_name_or_path: Path
    model_args: Optional[Any] = None
    kwargs: Optional[Any] = None


class HuggingFacePretrainedModel(NNModel):
    def __init__(
        self,
        model_type: HuggingFaceModelTypes,
        pretrained_model_name_or_path: Path,
        model_args: Optional[Any] = None,
        kwargs: Optional[Any] = None,
    ):
        super().__init__()
        if model_args is None:
            model_args = []
        if kwargs is None:
            kwargs = {}

        # TODO this would be perfect for a factory design, however the resovler register currently does not
        # support functions instead of classes within enums.
        self.huggingface_model = model_type.value.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.huggingface_model.forward(**inputs)

    @property
    def fsdp_block_names(self) -> List[str]:
        return self.huggingface_model._no_split_modules


def get_fsdp_block_from_huggingface_model(model: HuggingFacePretrainedModel) -> List[nn.Module]:
    fsdp_block_types = []
    for cls_block_name in model.fsdp_block_names:
        fsdp_block_types.append(FullyShardedDataParallelPlugin.get_module_class_from_name(model, cls_block_name))
    return fsdp_block_types


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
    model = HuggingFacePretrainedModel(
        model_type=HuggingFaceModelTypes.AutoModelForCausalLM, pretrained_model_name_or_path="epfl-llm/meditron-7b"
    )
    print(model)
    print(get_fsdp_block_from_huggingface_model(model))
