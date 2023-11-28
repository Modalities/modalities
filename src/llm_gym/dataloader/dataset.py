from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import jq
import numpy as np
from torch.utils.data.dataset import Dataset as TorchdataSet
from torch.utils.data.dataset import Subset
from transformers import GPT2TokenizerFast

from ..dataloader.large_file_lines_reader import LargeFileLinesReader


@dataclasses.dataclass
class DatasetSplit:
    train: Subset
    validation: Subset
    test: Subset


class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: str | Path, block_size: int):
        self.raw_data_path = Path(raw_data_path)
        self.block_size = block_size


class MemMapDataset(Dataset):
    def __init__(
        self, raw_data_path: str | Path, block_size: int, tokenizer_path: str | Path, jq_pattern: str = ".text"
    ):
        super().__init__(raw_data_path=raw_data_path, block_size=block_size)

        # if path is a dir, look for jsonl file
        if self.raw_data_path.is_dir():
            print(f"Data path '{self.raw_data_path}' is a directory, searching for .jsonl files...")
            files = list(self.raw_data_path.iterdir())
            files = [f for f in files if f.is_file()]
            files = [f for f in files if str(f).endswith(".jsonl")]
            if len(files) != 0:
                self.raw_data_path = files[0]
            else:
                raise ValueError(f"Could not detect any jsonl files in '{self.raw_data_path}'.")

        self.reader = LargeFileLinesReader(self.raw_data_path, lazy_init=True)
        self.jq_filter = jq.compile(jq_pattern)
        # TODO: tokenizer from tiktoken if it is faster?
        self.tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.reader)

    # TODO: tokenizer singleton?
    def __getitem__(self, idx: int) -> str:
        obj = self.tokenizer(
            self.jq_filter.input_text(self.reader[idx]).first(),
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
        )
        return obj


class PackedDataset(Dataset):
    def __init__(self, raw_data_path: str | Path, block_size: int, int_size_in_bytes: int = 4, max_samples: int = None):
        super().__init__(raw_data_path=raw_data_path, block_size=block_size)
        # if path is a dir, look for .packed.bin file
        if self.raw_data_path.is_dir():
            print(f"Data path '{self.raw_data_path}' is a directory, searching for .packed.bin files...")
            files = list(self.raw_data_path.iterdir())
            files = [f for f in files if f.is_file()]
            files = [f for f in files if str(f).endswith(".packed.bin")]
            if len(files) != 0:
                self.raw_data_path = files[0]
            else:
                raise ValueError(f"Could not detect any .packed.bin files in '{self.raw_data_path}'.")

        self.int_size_in_bytes = int_size_in_bytes

        # get number of total tokens in file
        with self.raw_data_path.open("r+b") as f:
            f.seek(0, os.SEEK_END)
            total_tokens = f.tell() // self.int_size_in_bytes
            f.seek(0)
        self.num_samples = total_tokens // self.block_size
        if max_samples:
            self.num_samples = min(self.num_samples, max_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        tokens_as_byte_strings = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=idx * self.int_size_in_bytes * self.block_size,
            shape=(self.int_size_in_bytes * self.block_size,),
        ).view(f"S{self.int_size_in_bytes}")
        tokens = [int.from_bytes(token, byteorder="big") for token in tokens_as_byte_strings]
        attention_mask = [1] * len(tokens)
        return {"input_ids": tokens, "attention_mask": attention_mask}
