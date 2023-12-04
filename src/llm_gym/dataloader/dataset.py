from __future__ import annotations

import dataclasses
import os
import pickle
from pathlib import Path
import jq
import numpy as np
from torch.utils.data.dataset import Dataset as TorchdataSet
from torch.utils.data.dataset import Subset
from tqdm import tqdm
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


# TODO: implement lazy data generation
# TODO: implement lazy index to file saving
class PackedMemMapDatasetBase(Dataset):
    INT_SIZE_IN_BYTES = 4

    def __init__(self, raw_data_path: str | Path, block_size: int):
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

        # get number of total bytes in file
        with self.raw_data_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            self.total_bytes = f.tell()
            f.seek(0)

        # get number of bytes in data section
        self.data_len = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=0,
            shape=(self.INT_SIZE_IN_BYTES,),
        ).view(f"S{self.INT_SIZE_IN_BYTES}")
        self.data_len = int.from_bytes(self.data_len, byteorder="big")

        # get index
        self.index_base = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=self.INT_SIZE_IN_BYTES + self.data_len,
            shape=(self.total_bytes - self.data_len - self.INT_SIZE_IN_BYTES,),
        ).view(f"S{self.total_bytes-self.data_len-self.INT_SIZE_IN_BYTES}")
        self.index_base = pickle.loads(self.index_base)


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    def __init__(self, raw_data_path: str | Path, block_size: int):
        super().__init__(raw_data_path=raw_data_path, block_size=block_size)

        # get number of total tokens in file
        total_tokens = self.data_len // self.INT_SIZE_IN_BYTES
        self.num_samples = total_tokens // self.block_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        tokens_as_byte_strings = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=self.INT_SIZE_IN_BYTES + idx * self.INT_SIZE_IN_BYTES * self.block_size,
            shape=(self.INT_SIZE_IN_BYTES * self.block_size,),
        ).view(f"S{self.INT_SIZE_IN_BYTES}")
        tokens = [int.from_bytes(token, byteorder="big") for token in tokens_as_byte_strings]
        attention_mask = [1] * len(tokens)
        return {"input_ids": tokens, "attention_mask": attention_mask}


class PackedMemMapDatasetMegatron(PackedMemMapDatasetBase):
    def generate_megatron_index(self):
        # index_output_path = Path(index_output_path)

        self.index = []
        curr_offset = 4
        curr_len = 0
        block_size_in_bytes = self.block_size * self.INT_SIZE_IN_BYTES
        for segment_offset, segment_len in tqdm(self.index_base):
            if curr_len + segment_len < block_size_in_bytes:
                curr_len += segment_len
            elif curr_len + segment_len == block_size_in_bytes:
                self.index.append((curr_offset, block_size_in_bytes))
                curr_len = 0
                curr_offset += block_size_in_bytes
            else:
                self.index.append((curr_offset, block_size_in_bytes))
                if segment_len > block_size_in_bytes:
                    curr_offset += block_size_in_bytes
                    curr_len = 0
                else:
                    curr_offset = segment_offset
                    curr_len = segment_len

    def __init__(self, raw_data_path: str | Path, block_size: int):
        super().__init__(raw_data_path=raw_data_path, block_size=block_size)

        # get number of total tokens in file
        total_tokens = self.data_len // self.INT_SIZE_IN_BYTES
        self.num_samples = total_tokens // self.block_size

        self.generate_megatron_index()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        offset, length = self.index[idx]
        tokens_as_byte_strings = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=offset,
            shape=(length,),
        ).view(f"S{self.INT_SIZE_IN_BYTES}")
        tokens = [int.from_bytes(token, byteorder="big") for token in tokens_as_byte_strings]
        attention_mask = [1] * len(tokens)
        return {"input_ids": tokens, "attention_mask": attention_mask}

