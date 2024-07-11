from abc import ABC
from typing import Dict, List, Optional

import sentencepiece as spm
from transformers import AutoTokenizer


class TokenizerWrapper(ABC):
    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, input_ids: List[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("Tokenizer must be implemented by a subclass.")

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError


class PreTrainedHFTokenizer(TokenizerWrapper):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        truncation: bool = False,
        padding: bool | str = False,
        max_length: Optional[int] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> None:
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
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def special_tokens(self) -> Dict[str, str | List[str]]:
        return self.tokenizer.special_tokens_map

    def tokenize(self, text: str) -> List[int]:
        tokens = self.tokenizer.__call__(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
        )["input_ids"]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        decoded_text = self.tokenizer.decode(token_ids)
        return decoded_text

    def get_token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, list):
            raise ValueError("Token is not represented by a single token id!")
        return token_id


class PreTrainedSPTokenizer(TokenizerWrapper):
    def __init__(self, tokenizer_model_file: str):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_model_file)
        pass

    def tokenize(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        decoded_text = self.tokenizer.decode(token_ids)
        return decoded_text

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

    def get_token_id(self, token: str) -> int:
        piece_id = self.tokenizer.PieceToId(token)
        if piece_id == self.tokenizer.unk_id():
            raise ValueError("Token is not represented by a single token id!")
        return piece_id
