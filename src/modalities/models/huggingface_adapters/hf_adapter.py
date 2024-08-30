import json
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from modalities.exceptions import ConfigError
from modalities.models.model import NNModel
from modalities.models.utils import ModelTypeEnum, get_model_from_config


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
        self, data_to_be_formatted: Union[Dict[str, Any], List[Any], PosixPath, Any]
    ) -> Union[Dict[str, Any], List[Any], PosixPath, Any]:
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
            Union[ModalitiesModelOutput, torch.Tensor]: The output of the forward pass.
        """
        # These parameters are required by HuggingFace. We do not use them and hence don't implement them.
        if output_attentions or output_hidden_states:
            raise NotImplementedError
        model_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        model_forward_output: Dict[str, torch.Tensor] = self.model.forward(model_input)
        if return_dict:
            return ModalitiesModelOutput(**model_forward_output)
        else:
            return model_forward_output[self.prediction_key]

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Prepares the inputs for generation.

        Args:
            input_ids (torch.LongTensor): The input tensor of token IDs.
            attention_mask (torch.LongTensor, optional): The attention mask tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the prepared inputs for generation.

        Note:
            Implement in subclasses of :class:`~transformers.PreTrainedModel`
            for custom behavior to prepare inputs in the generate method.
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


@dataclass
class ModalitiesModelOutput(ModelOutput):
    """
    ModalitiesModelOutput class.

    Args:
        logits (torch.FloatTensor, optional): The logits output of the model. Defaults to None.
        hidden_states (Tuple[torch.FloatTensor], optional): The hidden states output of the model. Defaults to None.
        attentions (Tuple[torch.FloatTensor], optional): The attentions output of the model. Defaults to None.
    """

    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
