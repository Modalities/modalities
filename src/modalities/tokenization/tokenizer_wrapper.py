from typing import List

import sentencepiece as spm
from transformers import PreTrainedTokenizer


class TokenizerWrapper:
    def tokenize(self, text: str):
        raise NotImplementedError("Tokenizer must be implemented by a subclass.")


class PreTrainedHFTokenizer(TokenizerWrapper):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, max_length: int, truncation: bool, padding: str = "max_length"
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.__call__(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
        )["input_ids"]


class PreTrainedSPTokenizer(TokenizerWrapper):
    def __init__(self, tokenizer: spm.SentencePieceProcessor = None):
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
