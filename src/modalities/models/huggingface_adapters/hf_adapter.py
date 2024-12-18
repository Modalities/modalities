import json
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Optional, Union, List, Tuple, Dict

import torch
from class_resolver.utils import logger
from torch import TensorType
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizer, \
    PreTrainedTokenizerFast, AddedToken
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput, TruncationStrategy
from transformers.utils import ModelOutput, PaddingStrategy

from modalities.exceptions import ConfigError
from modalities.models.model import NNModel
from modalities.models.utils import ModelTypeEnum, get_model_from_config, get_tokenizer_from_config

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}
SPIECE_UNDERLINE = "▁"


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


class HFTokenizerAdapter(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of #24622
            and #25224 which includes fixes to properly handle tokens that appear after special tokens.
            Make sure to also set `from_slow` to `True`.
            A simple example:

            - `legacy=True`:
            ```python
            # >>> from transformers import LlamaTokenizerFast
            #
            # >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=True, from_slow=True)
            # >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
            [1, 15043, 29871, 1, 869]
            ```
            - `legacy=False`:
            ```python
            # >>> from transformers import LlamaTokenizerFast
            #
            # >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
            # >>> tokenizer.encode("Hello <s>.")  # 29889 is '.'
            [1, 15043, 29871, 1, 29889]
            ```
            Checkout the [pull request](https://github.com/huggingface/transformers/pull/24565) for more details.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. Again, this should be set with `from_slow=True` to make sure it's taken into account.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            config: HFModelAdapterConfig,
            # vocab_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token=None,
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token=True,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            use_default_system_prompt=False,
            spaces_between_special_tokens=False,
            legacy=None,
            add_prefix_space=True,
            **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token

        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file"
                " you can ignore this message"
            )
            legacy = True

        self.legacy = legacy
        # self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.sp_model = get_tokenizer_from_config(config.config, "tokenizer")
        self.add_prefix_space = add_prefix_space

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    def unk_token_length(self):
        return len(self.sp_model.tokenizer.encode(str(self.unk_token)))

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state["sp_model"] = None
    #     state["sp_model_proto"] = self.sp_model.tokenizer.serialized_model_proto()
    #     return state
    #
    # def __setstate__(self, d):
    #     self.__dict__.update(d)
    #     self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
    #     self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.vocab_size

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the configuration
        config = HFModelAdapterConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create a new tokenizer instance
        tokenizer = cls(config=config, legacy=True, **kwargs)

        return tokenizer

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        text = text.replace(SPIECE_UNDERLINE, " ")
        if self.add_prefix_space:
            text = SPIECE_UNDERLINE + text

        tokens = super().tokenize(text, **kwargs)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return self.sp_model.tokenizer.encode(text, out_type=str)

            # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.tokenizer.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length:] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.tokenizer.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.tokenizer.IdToPiece(index)
        return token
