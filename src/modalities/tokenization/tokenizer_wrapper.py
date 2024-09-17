from abc import ABC
from typing import Dict, List, Optional

import sentencepiece as spm
from transformers import AutoTokenizer


class TokenizerWrapper(ABC):
    """Abstract interface for tokenizers."""

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes a text into a list of token IDs.

        Args:
            text (str): Text to be tokenized.

        Raises:
            NotImplementedError: Must be implemented by a subclass.

        Returns:
            List[int]: List of token IDs.
        """
        raise NotImplementedError

    def decode(self, input_ids: List[int]) -> str:
        """Decodes a list of token IDs into the original text.

        Args:
            input_ids (List[int]): List of token IDs.

        Raises:
            NotImplementedError: Must be implemented by a subclass.

        Returns:
            str: Decoded text.
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer.

        Raises:
            NotImplementedError: Must be implemented by a subclass.

        Returns:
            int: Vocabulary size.
        """
        raise NotImplementedError("Tokenizer must be implemented by a subclass.")

    def get_token_id(self, token: str) -> int:
        """Returns the token ID for a given token.

        Args:
            token (str): Token to get the ID for.

        Raises:
            NotImplementedError: Must be implemented by a subclass.

        Returns:
            int: Token ID.
        """
        raise NotImplementedError


class PreTrainedHFTokenizer(TokenizerWrapper):
    """Wrapper for pretrained Hugging Face tokenizers."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        truncation: Optional[bool] = False,
        padding: Optional[bool | str] = False,
        max_length: Optional[int] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes the PreTrainedHFTokenizer.

        Args:
            pretrained_model_name_or_path (str): Name or path of the pretrained model.
            truncation (bool, optional): Flag whether to apply truncation. Defaults to False.
            padding (bool | str, optional): Defines the padding strategy. Defaults to False.
            max_length (int, optional): Maximum length of the tokenization output. Defaults to None.
            special_tokens (Dict[str, str], optional): Added token keys should be in the list
                of predefined special attributes: [bos_token, eos_token, unk_token, sep_token, pad_token,
                cls_token, mask_token, additional_special_tokens].
                Example: {"pad_token": "[PAD]"}
                Tokens are only added if they are not already in the vocabulary (tested by checking
                if the tokenizer assign the index of the unk_token to them). Defaults to None.
        """
        # also see here for the truncation and padding options and their effects:
        # https://huggingface.co/docs/transformers/pad_truncation#padding-and-truncation

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        if special_tokens is not None:
            # TODO check if we always want to set
            # replace_additional_special_tokens=False
            self.tokenizer.add_special_tokens(
                special_tokens_dict=special_tokens,
                replace_additional_special_tokens=False,
            )
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer.

        Returns:
            int: Vocabulary size.
        """
        return self.tokenizer.vocab_size

    @property
    def special_tokens(self) -> Dict[str, str | List[str]]:
        """Returns the special tokens of the tokenizer.

        Returns:
            Dict[str, str | List[str]]: Special tokens dictionary.
        """
        return self.tokenizer.special_tokens_map

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes a text into a list of token IDs.

        Args:
            text (str): Text to be tokenized.

        Returns:
            List[int]: List of token IDs.
        """
        tokens = self.tokenizer.__call__(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
        )["input_ids"]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs into the original text.

        Args:
            input_ids (List[int]): List of token IDs.

        Returns:
            str: Decoded text.
        """
        decoded_text = self.tokenizer.decode(token_ids)
        return decoded_text

    def get_token_id(self, token: str) -> int:
        """Returns the token ID for a given token.

        Args:
            token (str): Token to get the ID for.

        Raises:
            ValueError: If the token cannot be represented by a single token ID.

        Returns:
            int: Token ID.
        """
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, list):
            raise ValueError("Token is not represented by a single token id!")
        return token_id


class PreTrainedSPTokenizer(TokenizerWrapper):
    """Wrapper for pretrained SentencePiece tokenizers."""

    def __init__(self, tokenizer_model_file: str):
        """Initializes the PreTrainedSPTokenizer.

        Args:
            tokenizer_model_file (str): Path to the tokenizer model file.
        """
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_model_file)

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes a text into a list of token IDs.

        Args:
            text (str): Text to be tokenized.

        Returns:
            List[int]: List of token IDs.
        """
        tokens = self.tokenizer.encode(text)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs into the original text.

        Args:
            input_ids (List[int]): List of token IDs.

        Returns:
            str: Decoded text.
        """
        decoded_text = self.tokenizer.decode(token_ids)
        return decoded_text

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer.

        Returns:
            int: Vocabulary size.
        """
        return self.tokenizer.vocab_size()

    def get_token_id(self, token: str) -> int:
        """Returns the token ID for a given token.

        Args:
            token (str): Token to get the ID for.

        Raises:
            ValueError: If the token cannot be represented by a single token ID.

        Returns:
            int: Token ID.
        """
        piece_id = self.tokenizer.PieceToId(token)
        if piece_id == self.tokenizer.unk_id():
            raise ValueError("Token is not represented by a single token id!")
        return piece_id
