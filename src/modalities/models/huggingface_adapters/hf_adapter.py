import json
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Optional, Union, List, Tuple

import torch
from torch import TensorType
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizer
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput, TruncationStrategy
from transformers.utils import ModelOutput, PaddingStrategy

from modalities.exceptions import ConfigError
from modalities.models.model import NNModel
from modalities.models.utils import ModelTypeEnum, get_model_from_config, get_tokenizer_from_config


class HFModelAdapterConfig(PretrainedConfig):
    """HFModelAdapterConfig configuration class for the HFModelAdapter."""

    model_type = "modalities"

    def __init__(self, **kwargs):
        """
        Initializes an HFModelAdapterConfig object.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            ConfigError: If the config is not passed in HFModelAdapterConfig.
        """
        super().__init__(**kwargs)
        # self.config is added by the super class via kwargs
        if self.config is None:
            raise ConfigError("Config is not passed in HFModelAdapterConfig.")
        # since the config will be saved to json and json can't handle posixpaths, we need to convert them to strings
        self._convert_posixpath_to_str(data_to_be_formatted=self.config)

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Converts the adapter object configuration to a JSON string representation.

        Args:
            use_diff (bool, optional): Whether to include only the differences from the default configuration.
                Defaults to True.

        Returns:
            str: The JSON string representation of the adapter object.
        """
        json_dict = {"config": self.config.copy(), "model_type": self.model_type}
        return json.dumps(json_dict)

    def _convert_posixpath_to_str(
        self, data_to_be_formatted: dict[str, Any] | list[Any] | PosixPath | Any
    ) -> dict[str, Any] | list[Any] | PosixPath | Any:
        # Recursively converts any PosixPath objects within a nested data structure to strings.

        if isinstance(data_to_be_formatted, dict):
            for key, value in data_to_be_formatted.items():
                data_to_be_formatted[key] = self._convert_posixpath_to_str(data_to_be_formatted=value)
        elif isinstance(data_to_be_formatted, list):
            for i in range(len(data_to_be_formatted)):
                data_to_be_formatted[i] = self._convert_posixpath_to_str(data_to_be_formatted=data_to_be_formatted[i])
        elif isinstance(data_to_be_formatted, PosixPath):
            return str(data_to_be_formatted)
        return data_to_be_formatted


class HFModelAdapter(PreTrainedModel):
    """HFModelAdapter class for the HuggingFace model adapter."""

    config_class = HFModelAdapterConfig

    def __init__(
        self, config: HFModelAdapterConfig, prediction_key: str, load_checkpoint: bool = False, *inputs, **kwargs
    ):
        """
        Initializes the HFAdapter object.

        Args:
            config (HFModelAdapterConfig): The configuration object for the HFAdapter.
            prediction_key (str): The key for the prediction.
            load_checkpoint (bool, optional): Whether to load a checkpoint. Defaults to False.
            *inputs: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(config, *inputs, **kwargs)
        self.prediction_key = prediction_key
        if load_checkpoint:
            self.model: NNModel = get_model_from_config(config.config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
        else:
            self.model: NNModel = get_model_from_config(config.config, model_type=ModelTypeEnum.MODEL)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ):
        """
        Forward pass of the HFAdapter module.

        Args:
            input_ids (torch.Tensor): The input tensor of token indices.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary as output. Defaults to False.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to False.

        Returns:
            ModalitiesModelOutput | torch.Tensor: The output of the forward pass.
        """
        # These parameters are required by HuggingFace. We do not use them and hence don't implement them.
        if output_attentions or output_hidden_states:
            raise NotImplementedError
        model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        model_forward_output: dict[str, torch.Tensor] = self.model.forward(model_input)
        if return_dict:
            return ModalitiesModelOutput(**model_forward_output)
        else:
            return model_forward_output[self.prediction_key]

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None, **kwargs
    ) -> dict[str, Any]:
        """
        Prepares the inputs for generation.

        Args:
            input_ids (torch.LongTensor): The input tensor of token IDs.
            attention_mask (torch.LongTensor, optional): The attention mask tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the prepared inputs for generation.

        Note:
            Implement in subclasses of :class:`~transformers.PreTrainedModel`
            for custom behavior to prepare inputs in the generate method.
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class HFTokenizerAdapter(PreTrainedTokenizer):
    # config_class = HFModelAdapterConfig
    def __init__(
            self, config: HFModelAdapterConfig, **kwargs
    ):
        super().__init__(**kwargs)

        self.tokenizer = get_tokenizer_from_config(config.config, "tokenizer")
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_token = config.eos_token if hasattr(config, 'eos_token') else "</s>"
        self.bos_token = config.bos_token if hasattr(config, 'bos_token') else "<s>"
        self.pad_token = config.pad_token if hasattr(config, 'pad_token') else "<pad>"
        self.init_kwargs = kwargs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the configuration
        config = HFModelAdapterConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create a new tokenizer instance
        tokenizer = cls(config, **kwargs)

        return tokenizer

    def encode(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            stride: int = 0,
            return_tensors: Optional[Union[str, TensorType]] = None,
            **kwargs,
    ) -> List[int]:
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, token):
        return self.tokenizer.get_token_id(token=token)

    def decode(
            self,
            token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
    ) -> str:
        return self.tokenizer.decode(token_ids)



@dataclass
class ModalitiesModelOutput(ModelOutput):
    """
    ModalitiesModelOutput class.

    Args:
        logits (torch.FloatTensor, optional): The logits output of the model. Defaults to None.
        hidden_states (tuple[torch.FloatTensor], optional): The hidden states output of the model. Defaults to None.
        attentions (tuple[torch.FloatTensor], optional): The attentions output of the model. Defaults to None.
    """

    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
