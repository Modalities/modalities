from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, ConfigDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    LongT5ForConditionalGeneration,
    LongT5Model,
)

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
    LongT5Model = LongT5Model
    LongT5ForConditionalGeneration = LongT5ForConditionalGeneration


class HuggingFacePretrainedModelConfig(BaseModel):
    model_type: HuggingFaceModelTypes
    model_name: Path
    prediction_key: str
    huggingface_prediction_subscription_key: str
    sample_key: str
    model_args: Optional[Any] = None
    kwargs: Optional[Any] = None

    # avoid warning about protected namespace 'model_', see
    # https://docs.pydantic.dev/2.7/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())


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
        if "llama" in str(model_name).lower():
            weight_decay_groups = {
                "linear": [".self_attn", ".mlp"],
                "embedding": [".embed_tokens", ".lm_head"],
                "layernorm": [".norm"],
            }
        else:
            # must set weight_decay_groups_excluded in config file to empty list
            weight_decay_groups = {}

        super().__init__(weight_decay_groups=weight_decay_groups)
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


class HuggingFacePretrainedEncoderDecoderModel(NNModel):
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
        assert model_type.value in [LongT5Model, LongT5ForConditionalGeneration]
        if model_type.value in [LongT5Model, LongT5ForConditionalGeneration]:
            # with the regex, we match all parameters in the SelfAttention layer,
            # except global_input_layer_norm to prevent double counting
            weight_decay_groups = {
                "linear": [
                    ".DenseReluDense\\.w",
                    ".SelfAttention\\.([^g]|global_)($|[^i])",
                    ".EncDecAttention",
                    ".lm_head",
                ],
                "embedding": [".shared", ".embed_tokens"],
                "layernorm": [".layer_norm"],
            }
        else:
            # must set weight_decay_groups_excluded in config file to empty list
            weight_decay_groups = {}

        super().__init__(weight_decay_groups=weight_decay_groups)
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

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # TODO: refactor so that target_key and decoder_start_token_id can be set in config/obtained automatically
        decoder_input_ids = self._shift_tokens_right(targets["target_ids"], 1)
        output = self.huggingface_model.forward(
            input_ids=inputs[self.sample_key],
            decoder_input_ids=decoder_input_ids,
        )
        return {self.prediction_key: output[self.huggingface_prediction_subscription_key]}

    @staticmethod
    def _shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int) -> torch.Tensor:
        """Shifts target sequence right by 1 to create decoder input sequence,
        which is used for autoregressive modeling with teacher forcing.
        LongT5 accepts either decoder_inputs or targets and _shift_tokens_right is already implemented by LongT5,
        however, when passing targets it also already computes the loss and this fails due to type mismatch.
        Computing it here is a workaround.

        Args:
            input_ids (torch.Tensor): _description_
            decoder_start_token_id (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        shifted_input_ids = torch.roll(input_ids, 1, dims=1)
        shifted_input_ids[:, 0] = decoder_start_token_id
        return shifted_input_ids

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
