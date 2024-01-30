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
from .create_packed_data import PackedDataGenerator


class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        self.raw_data_path = raw_data_path
        self.block_size = block_size
        self.sample_key = sample_key

    def _check_if_inbounds(self, idx: int):
        if not 0 <= idx < len(self):
            raise IndexError


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
        Pytorch Dataset with mmap support.

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
        self._check_if_inbounds(idx)
        return self.tokenizer(
            self.jq_filter.input_text(self.reader[idx]).first(),
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
        )


class PackedMemMapDatasetBase(Dataset):
    DATA_SECTION_LENGTH_IN_BYTES = PackedDataGenerator.DATA_SECTION_LENGTH_IN_BYTES
    TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES = PackedDataGenerator.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES
    HEADER_SIZE_IN_BYTES = PackedDataGenerator.HEADER_SIZE_IN_BYTES
    np_dtype_from_num_bytes = {
        1: np.dtype(np.uint8).newbyteorder(">"),
        2: np.dtype(np.uint16).newbyteorder(">"),
        4: np.dtype(np.uint32).newbyteorder(">"),
        8: np.dtype(np.uint64).newbyteorder(">"),
    }

    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        """
        Base class for packed memmapped datasets. The underlying dataset file has the structure:
        | header | data | index |
        The header contains information about the length of the subsequent data sequence and the amount of bytes
        required to represent tokens in the data section. The index contains the tuple information (start, end) in terms
         of byte positions.

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

        with self.raw_data_path.open("rb") as f:
            # get number of total bytes in file
            f.seek(0, os.SEEK_END)
            self.total_bytes = f.tell()
            f.seek(0)

            # get number of bytes in data section
            data_section_length_in_bytes = f.read(self.DATA_SECTION_LENGTH_IN_BYTES)
            self.data_len = int.from_bytes(data_section_length_in_bytes, byteorder="big")

            # get number of bytes for encoding a single token
            f.seek(self.DATA_SECTION_LENGTH_IN_BYTES)
            token_size_as_bytes = f.read(self.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES)
            self._token_size_in_bytes = int.from_bytes(token_size_as_bytes, byteorder="big", signed=False)
            try:
                self._token_dtype = self.np_dtype_from_num_bytes[self._token_size_in_bytes]
            except KeyError:
                raise RuntimeError(
                    f"Encountered a required token representation with {self._token_size_in_bytes},"
                    " which is not supported. Consider using a smaller vocabulary."
                )

            # get index
            f.seek(self.HEADER_SIZE_IN_BYTES + self.data_len)
            pkl_encoded_index = f.read()
            self.index_base = pickle.loads(pkl_encoded_index)

            # initialize memmapped data section
            self.data = np.memmap(
                self.raw_data_path, mode="r", offset=self.HEADER_SIZE_IN_BYTES, shape=(self.data_len,)
            )

            self._index = self._generate_packing_index()

    def _generate_packing_index(self) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> BatchEncoding:
        self._check_if_inbounds(idx)
        offset, length = self._index[idx]
        tokens = np.frombuffer(self.data, dtype=self._token_dtype, count=length, offset=offset)
        return BatchEncoding(data={self.sample_key: tokens})


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    def _generate_packing_index(self) -> List[Tuple[int, int]]:
        # get number of total tokens in file
        total_tokens = self.data_len // self._token_size_in_bytes
        num_samples = total_tokens // self.block_size
        return [(i * self.block_size * self._token_size_in_bytes, self.block_size) for i in range(num_samples)]


class PackedMemMapDatasetMegatron(PackedMemMapDatasetBase):
    def _generate_packing_index(self) -> List[Tuple[int, int]]:
        index = []
        curr_offset = self.HEADER_SIZE_IN_BYTES
        curr_len = 0
        block_size_in_bytes = self.block_size * self._token_size_in_bytes
        for segment_offset, segment_len in tqdm(self.index_base):
            # When the sum of the length of the current previously seen samples doesn't
            # exceed block_size_in_bytes, we add the current segment length to the previous
            # ones and continue.
            if curr_len + segment_len < block_size_in_bytes:
                curr_len += segment_len
            # If the previous and current length equals block_size_in_bytes, we add the starting index
            # and the total sequences length to the index list as a new sample.
            elif curr_len + segment_len == block_size_in_bytes:
                index.append((curr_offset, self.block_size))
                curr_len = 0
                curr_offset += block_size_in_bytes
            # Else case is executed when the current and previous segment length exceed the block_size.
            # In this case we set the starting point of the next sample to the end of the current sample.
            # This way, the start of a sample is never in the middle of a sentence.
            else:
                index.append((curr_offset, self.block_size))
                if segment_len > block_size_in_bytes:
                    curr_offset += block_size_in_bytes
                    curr_len = 0
                else:
                    curr_offset = segment_offset
                    curr_len = segment_len
        return index
