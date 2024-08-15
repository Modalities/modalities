from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

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
    """
    HuggingFaceModelTypes enumeration class representing different types of HuggingFace models.

    Attributes:
        AutoModelForCausalLM: Represents the AutoModelForCausalLM class.
        AutoModelForMaskedLM: Represents the AutoModelForMaskedLM class.
    """

    AutoModelForCausalLM = AutoModelForCausalLM
    AutoModelForMaskedLM = AutoModelForMaskedLM


class HuggingFacePretrainedModelConfig(BaseModel):
    """
    Configuration class for HuggingFacePretrainedModel.

    Attributes:
        model_type (HuggingFaceModelTypes): The type of the HuggingFace model.
        model_name (Path): The path to the HuggingFace model.
        prediction_key (str): The key for accessing the prediction.
        huggingface_prediction_subscription_key (str): The subscription key for HuggingFace prediction.
        sample_key (str): The key for accessing the sample.
        model_args (Any, optional): Optional additional arguments for the model.
        kwargs (Any, optional): Optional additional keyword arguments.
    """

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
    """HuggingFacePretrainedModel class for HuggingFace models."""

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
        """
        Initializes a HuggingFaceModel object.

        Args:
            model_type (HuggingFaceModelTypes): The type of Hugging Face model.
            model_name (str): The name of the Hugging Face model.
            prediction_key (str): The key for accessing predictions.
            huggingface_prediction_subscription_key (str): The subscription key for Hugging Face predictions.
            sample_key (str): The key for accessing samples.
            model_args (Any, optional): Additional arguments for the Hugging Face model. Defaults to None.
            kwargs (Any, optional): Additional keyword arguments for the Hugging Face model. Defaults to None.
        """
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
        """
        Forward pass of the model.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing output tensors.
        """
        output = self.huggingface_model.forward(inputs[self.sample_key])
        return {self.prediction_key: output[self.huggingface_prediction_subscription_key]}

    @property
    def fsdp_block_names(self) -> List[str]:
        """
        Returns a list of FSDP block names.

        Returns:
            List[str]: A list of FSDP block names.
        """
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
