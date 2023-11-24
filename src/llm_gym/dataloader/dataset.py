from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Iterable, Type, Union

import jq
import numpy as np
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset as TorchdataSet
from torch.utils.data.dataset import Subset
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

from ..dataloader.large_file_lines_reader import LargeFileLinesReader


@dataclasses.dataclass
class DatasetSplit:
    train: Subset
    validation: Subset
    test: Subset


class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: Union[str, Path]):
        self.raw_data_path = Path(raw_data_path)

    @staticmethod
    def from_path(
        dataset_path: str, target_dataset_cls: Type[Dataset], split_size: Iterable[float] = (0.9, 0.05, 0.05), **kwargs
    ) -> DatasetSplit:
        presplit_dataset_folder_paths = [Path(dataset_path, split) for split in ["train", "test", "validation"]]

        def get_subset(ds):
            return Subset(dataset=ds, indices=range(len(ds)))

        if all(p.is_dir() for p in presplit_dataset_folder_paths):
            print(f"Found already existing dataset split at {dataset_path}. Will use this one...")

            def init_dataset(path, **kwargs):
                return target_dataset_cls(raw_data_path=path, **kwargs)

            dataset_split = [init_dataset(p, **kwargs) for p in presplit_dataset_folder_paths]
        else:
            print(f"No existing dataset split found at {dataset_path}. Loading dataset directly and apply split")
            dataset = target_dataset_cls(raw_data_path=dataset_path, **kwargs)
            dataset_split = random_split(dataset, split_size)
        return DatasetSplit(
            train=get_subset(dataset_split[0]),
            validation=get_subset(dataset_split[1]),
            test=get_subset(dataset_split[2]),
        )


class MemMapDataset(Dataset):
    def __init__(
        self,
        raw_data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerFast = GPT2TokenizerFast(tokenizer_file="./data/tokenizer/tokenizer.json"),
        jq_pattern: str = ".text",
    ):
        super().__init__(raw_data_path=raw_data_path)

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
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.reader)

    # TODO: tokenizer singleton?
    def __getitem__(self, idx: int) -> str:
        obj = self.tokenizer(
            self.jq_filter.input_text(self.reader[idx]).first(), max_length=1024, padding="max_length", truncation=True
        )
        return obj


class PackedDataset(Dataset):
    def __init__(
        self, raw_data_path: str | Path, block_size: int = 1024, int_size_in_bytes: int = 4, max_samples: int = None
    ):
        super().__init__(raw_data_path=raw_data_path)
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

        self.block_size = block_size
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
