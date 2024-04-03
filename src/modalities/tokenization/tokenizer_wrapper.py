from typing import List

import sentencepiece as spm
from transformers import AutoTokenizer


class TokenizerWrapper:
    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("Tokenizer must be implemented by a subclass.")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("Tokenizer must be implemented by a subclass.")


class PreTrainedHFTokenizer(TokenizerWrapper):
    def __init__(
        self, pretrained_model_name_or_path: str, max_length: int, truncation: bool = True, padding: str = "max_length"
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.__call__(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
        )["input_ids"]


class PreTrainedSPTokenizer(TokenizerWrapper):
    def __init__(self, tokenizer_model_file: str):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_model_file)

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()
