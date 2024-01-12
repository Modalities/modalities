from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import jq
import numpy as np
from torch.utils.data.dataset import Dataset as TorchdataSet
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer

from ..dataloader.large_file_lines_reader import LargeFileLinesReader


class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        self.raw_data_path = raw_data_path
        self.block_size = block_size
        self.sample_key = sample_key


class MemMapDataset(Dataset):
    def __init__(
        self,
        raw_data_path: Path,
        block_size: int,
        tokenizer: PreTrainedTokenizer,
        sample_key: str,
        index_path: Optional[Path] = None,
        jq_pattern: str = ".text",
    ):
        """
        :param raw_data_path: Path to a jsonl file, which holds text data
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param tokenizer: PretrainedTokenizer required to tokenize text data on the fly.
        :param jq_pattern: jq-pattern applied on every jsonl-entry. Results are afterwards tokenized and packed
        :param index_path: Path to an index file, which indicates the start character/byte position
                           and length of samples given in `raw_data_path`.
                           If not defined, an index next to `raw_data_path` is picked,
                           by replacing its suffix with ".idx".
        :param sample_key: model-specific parameter to indicate where in the BatchEncoding the input_token_ids are.
                           TODO: If this setting should support multi-modal features using separately encoded inputs,
                            this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key)

        self.reader = LargeFileLinesReader(self.raw_data_path, index_path=index_path)
        self.jq_filter = jq.compile(jq_pattern)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> BatchEncoding:
        return self.tokenizer(
            self.jq_filter.input_text(self.reader[idx]).first(),
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
        )


class PackedMemMapDatasetBase(Dataset):
    INT_SIZE_IN_BYTES = 4
    HEADER_SIZE_IN_BYTES = 8

    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        """
        Base class for packed memmapped datasets. The underlying dataset file has the structure:
        | header | data | index |
        The header contains information about the length of the subsequent data sequence. The index contains
        the tuple information (start, end) in terms of byte positions.

        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `modalities create_packed_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param sample_key: model-specific parameter to indicate where in the BatchEncoding the input_token_ids are.
                           TODO: If this setting should support multi-modal features using separately encoded inputs,
                            this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key)
        if not self.raw_data_path.is_file():
            raise FileNotFoundError(
                f"Packed Data was not found at {self.raw_data_path}."
                f"Create on in advance by using `modalities create_packed_data`."
            )

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
            shape=(self.HEADER_SIZE_IN_BYTES,),
        ).view(f"S{self.HEADER_SIZE_IN_BYTES}")
        self.data_len = int.from_bytes(self.data_len, byteorder="big")

        # get index
        self.index_base = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=self.HEADER_SIZE_IN_BYTES + self.data_len,
            shape=(self.total_bytes - self.data_len - self.HEADER_SIZE_IN_BYTES,),
        ).view(f"S{self.total_bytes-self.data_len-self.HEADER_SIZE_IN_BYTES}")
        self.index_base = pickle.loads(self.index_base)


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        """
        PackedMemMapDatasetContinuous iterates through the data in block_size sized chunks,
        irrespective of the samples' start and end position, as defined in the index.

        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `modalities create_packed_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param sample_key: model-specific parameter to indicate where in the BatchEncoding the input_token_ids are.
                           TODO: If this setting should support multi-modal features using separately encoded inputs,
                            this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key)

        # get number of total tokens in file
        total_tokens = self.data_len // self.INT_SIZE_IN_BYTES
        self._num_samples = total_tokens // self.block_size

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> BatchEncoding:
        tokens_as_byte_strings = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=self.HEADER_SIZE_IN_BYTES + idx * self.INT_SIZE_IN_BYTES * self.block_size,
            shape=(self.INT_SIZE_IN_BYTES * self.block_size,),
        ).view(f"S{self.INT_SIZE_IN_BYTES}")
        tokens = [int.from_bytes(token, byteorder="big") for token in tokens_as_byte_strings]
        return BatchEncoding(data={self.sample_key: tokens})


class PackedMemMapDatasetMegatron(PackedMemMapDatasetBase):
    def generate_megatron_index(self) -> List[Tuple[int, int]]:
        index = []
        curr_offset = self.HEADER_SIZE_IN_BYTES
        curr_len = 0
        block_size_in_bytes = self.block_size * self.INT_SIZE_IN_BYTES
        for segment_offset, segment_len in tqdm(self.index_base):
            # When the sum of of the length of the current previously seen samples doesn't
            # exceed block_size_in_bytes, we add the current segment length to the previous
            # ones and continue.
            if curr_len + segment_len < block_size_in_bytes:
                curr_len += segment_len
            # If the previous and current length equals block_size_in_bytes, we add the starting index
            # and the total sequences length to the index list as a new sample.
            elif curr_len + segment_len == block_size_in_bytes:
                index.append((curr_offset, block_size_in_bytes))
                curr_len = 0
                curr_offset += block_size_in_bytes
            # Else case is executed when the current and previous segment length exceed the block_size.
            # In this case we set the starting point of the next sample to the end of the current sample.
            # This way, the start of a sample is never in the middle of a sentence.
            else:
                index.append((curr_offset, block_size_in_bytes))
                if segment_len > block_size_in_bytes:
                    curr_offset += block_size_in_bytes
                    curr_len = 0
                else:
                    curr_offset = segment_offset
                    curr_len = segment_len
        return index

    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        """
        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `modalities create_packed_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param sample_key: model-specific parameter to indicate where in the BatchEncoding the input_token_ids are.
                           TODO: If this setting should support multi-modal features using separately encoded inputs,
                            this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key)
        self._index = self.generate_megatron_index()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> BatchEncoding:
        offset, length = self._index[idx]
        tokens_as_byte_strings = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=offset,
            shape=(length,),
        ).view(f"S{self.INT_SIZE_IN_BYTES}")
        tokens = [int.from_bytes(token, byteorder="big") for token in tokens_as_byte_strings]
        return BatchEncoding(data={self.sample_key: tokens})
