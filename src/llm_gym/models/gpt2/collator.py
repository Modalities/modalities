from dataclasses import field
from pathlib import Path
from typing import List

import torch
from transformers import DataCollatorForLanguageModeling, GPT2TokenizerFast

from llm_gym.batch import DatasetBatch


class Tokenizer(GPT2TokenizerFast):
    DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent.parent / Path("data", "tokenizer", "tokenizer.json")

    # Introducing another abstraction for a tokenizer, in case tiktoken is fast than `transformers.GPT2TokenizerFast`
    @property
    def pad_token(self) -> str:
        return self.eos_token


class GPT2LLMCollator:
    def __init__(self, target_publication_key: str, pad_to_multiple_of: int = 8):
        self.device: torch.device = field(default_factory=lambda: torch.device("cpu"))
        self.target_publication_key = target_publication_key
        tokenizer = Tokenizer(Tokenizer.DEFAULT_TOKENIZER_PATH)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, batch: List[torch.Tensor]) -> DatasetBatch:
        """
        :param batch: batch format [no_samples, height, width, channels]
        :return:
        """
        collated_batch = self.data_collator(batch)
        samples = {"input_ids": collated_batch["input_ids"], "attention_mask": collated_batch["attention_mask"]}
        targets = {
            self.target_publication_key: collated_batch["labels"],
            "attention_mask": collated_batch["attention_mask"],
        }
        return DatasetBatch(targets=targets, samples=samples)
