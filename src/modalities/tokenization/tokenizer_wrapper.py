import warnings
from abc import ABC
from typing import Optional

import sentencepiece as spm
from transformers import AutoTokenizer


class TokenizerWrapper(ABC):
    """Abstract interface for tokenizers."""

    def tokenize(self, text: str) -> list[int]:
        """Tokenizes a text into a list of token IDs.

        Args:
            text (str): Text to be tokenized.

        Raises:
            NotImplementedError: Must be implemented by a subclass.

        Returns:
            list[int]: List of token IDs.
        """
        raise NotImplementedError

    def decode(self, input_ids: list[int]) -> str:
        """Decodes a list of token IDs into the original text.

        Args:
            input_ids (list[int]): List of token IDs.

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

    def is_special_token_id(self, token_id: int) -> bool:
        """Returns whether a token ID is a special token ID.

        Args:
            token_id (int): Token ID to check.

        Raises:
            NotImplementedError: Must be implemented by a subclass.

        Returns:
            bool: Flag whether the token ID is a special token ID.
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
        special_tokens: Optional[dict[str, str]] = None,
    ) -> None:
        """Initializes the PreTrainedHFTokenizer.

        Args:
            pretrained_model_name_or_path (str): Name or path of the pretrained model.
            truncation (bool, optional): Flag whether to apply truncation. Defaults to False.
            padding (bool | str, optional): Defines the padding strategy. Defaults to False.
            max_length (int, optional): Maximum length of the tokenization output. Defaults to None.
            special_tokens (dict[str, str], optional): Added token keys should be in the list
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
            old_vocab_size = len(self.tokenizer.get_vocab())
            # TODO check if we always want to set
            # replace_additional_special_tokens=False
            self.tokenizer.add_special_tokens(
                special_tokens_dict=special_tokens,
                replace_additional_special_tokens=False,
            )
            if len(self.tokenizer.get_vocab()) > old_vocab_size:
                raise NotImplementedError(
                    "Currently only tokens already known to the tokenizers vocabulary can be added,"
                    + " as resizing the embedding matrix is not yet supported!"
                )
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.special_token_ids = set(self.tokenizer.all_special_ids)

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer.

        Returns:
            int: Vocabulary size.
        """
        return self.tokenizer.vocab_size

    @property
    def special_tokens(self) -> dict[str, str | list[str]]:
        """Returns the special tokens of the tokenizer.

        Returns:
            dict[str, str | list[str]]: Special tokens dictionary.
        """
        return self.tokenizer.special_tokens_map

    def tokenize(self, text: str) -> list[int]:
        """Tokenizes a text into a list of token IDs.

        Args:
            text (str): Text to be tokenized.

        Returns:
            list[int]: List of token IDs.
        """
        tokens = self.tokenizer.__call__(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
        )["input_ids"]
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decodes a list of token IDs into the original text.

        Args:
            input_ids (list[int]): List of token IDs.

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
        if not isinstance(token_id, int):
            raise ValueError("Token is not represented by a single token id!")
        if token_id is None:
            raise ValueError("Token is not represented by a single token id!")
        elif token_id == self.tokenizer.unk_token_id:
            warnings.warn(f"The provided eod token {token} has the same token id ({token_id}) as the unk token")
        return token_id

    def is_special_token_id(self, token_id: int) -> bool:
        """Returns whether a token ID is a special token ID.

        Args:
            token_id (int): Token ID to check.

        Returns:
            bool: Flag whether the token ID is a special token ID.
        """
        return token_id in self.special_token_ids


class PreTrainedSPTokenizer(TokenizerWrapper):
    """Wrapper for pretrained SentencePiece tokenizers."""

    def __init__(self, tokenizer_model_file: str):
        """Initializes the PreTrainedSPTokenizer.

        Args:
            tokenizer_model_file (str): Path to the tokenizer model file.
        """
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_model_file)

    def tokenize(self, text: str) -> list[int]:
        """Tokenizes a text into a list of token IDs.

        Args:
            text (str): Text to be tokenized.

        Returns:
            list[int]: List of token IDs.
        """
        token_ids = self.tokenizer.Encode(text)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decodes a list of token IDs into the original text.

        Args:
            input_ids (list[int]): List of token IDs.

        Returns:
            str: Decoded text.
        """
        decoded_text = self.tokenizer.Decode(token_ids)
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
        if not isinstance(piece_id, int):
            raise ValueError("Token cannot be represented by a single token ID!")
        if piece_id == self.tokenizer.unk_id():
            raise ValueError("Token  cannot be represented by a single token id!")
        return piece_id

    def is_special_token_id(self, token_id: int) -> bool:
        """Returns whether a token ID is a special token ID.

        Args:
            token_id (int): Token ID to check.

        Raises:
            NotImplementedError: Must be implemented by a subclass.

        Returns:
            bool: Flag whether the token ID is a special token ID.
        """
        return self.tokenizer.IsControl(token_id)
