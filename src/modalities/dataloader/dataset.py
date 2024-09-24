from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import jq
import numpy as np
from pydantic import BaseModel
from torch.utils.data.dataset import Dataset as TorchdataSet
from tqdm import tqdm
from transformers import BatchEncoding

from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper

from ..dataloader.large_file_lines_reader import LargeFileLinesReader
from .create_packed_data import EmbeddedStreamData


class Dataset(TorchdataSet):
    """Dataset class."""

    def __init__(self, raw_data_path: Path, sample_key: str):
        """
        Initializes a Dataset object.

        Args:
            raw_data_path (Path): The path to the raw data.
            sample_key (str): The key used to access a sample in the dataset.
        """
        self.raw_data_path = raw_data_path
        self.sample_key = sample_key

    def _check_if_inbounds(self, idx: int):
        # check if the provided index is within the bounds of the dataset.
        if not 0 <= idx < len(self):
            raise IndexError


class DummySampleDataType(str, Enum):
    """
    DummySampleDataType is an enumeration class that represents the data types for dummy samples.

    Attributes:
        FLOAT (str): Represents the float data type.
        INT (str): Represents the int data type.
    """

    FLOAT = "float"
    INT = "int"


class DummySampleConfig(BaseModel):
    """
    DummySampleConfig class represents the configuration for a dummy sample.

    Attributes:
        sample_key (str): The key of the sample.
        sample_shape (tuple[int, ...]): The shape of the sample.
        sample_type (DummySampleDataType): The type of the sample.

    """

    sample_key: str
    sample_shape: tuple[int, ...]
    sample_type: DummySampleDataType


class DummyDatasetConfig(BaseModel):
    """
    DummyDatasetConfig is a configuration class for defining a dummy dataset.

    Attributes:
        num_samples (int): The number of samples in the dataset.
        sample_definition (list[DummySampleConfig]): The list of sample definitions in the dataset.
    """

    num_samples: int
    sample_definition: list[DummySampleConfig]


class DummyDataset(Dataset):
    """DummyDataset class."""

    def __init__(self, num_samples: int, sample_definition: tuple[DummySampleConfig]):
        """
        Initializes a DummyDataset object with the given number of samples and sample definition.
        When calling the __getitem__ method, the dataset will return a random sample based on the sample definition.

        Args:
            num_samples (int): The number of samples in the dataset.
            sample_definition (tuple[DummySampleConfig]): A list of tuples defining the dataset output.
                Each touple contains the sample key, shape and data type.

        Returns:
            None
        """
        super().__init__(raw_data_path=None, sample_key=None)
        self.num_samples = num_samples
        self.sample_definition = sample_definition

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves an item from the dataset at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary representing the retrieved item.

        Note:
            idx is not used. Instedam the method returns a random sample.
        """
        return self._create_random_sample()

    def _create_random_sample(self) -> dict:
        # creates a random sample based on the sample definition
        sample = dict()
        for s in self.sample_definition:
            if s.sample_type == DummySampleDataType.FLOAT:
                data = np.random.randn(*s.sample_shape)
            elif s.sample_type == DummySampleDataType.INT:
                data = np.random.randint(low=0, high=512, size=s.sample_shape)
            else:
                raise NotImplementedError(f"DummyDataset does not support type { s.sample_type}")
            sample[s.sample_key] = data
        return sample


class MemMapDataset(Dataset):
    def __init__(
        self,
        raw_data_path: Path,
        tokenizer: TokenizerWrapper,
        sample_key: str,
        index_path: Optional[Path] = None,
        jq_pattern: str = ".text",
    ):
        """
        Initializes the MemMapDataset object that represents a PyTorch Dataset with mmap support.

        Args:
            raw_data_path (Path): Path to a JSONL file, which holds text data.
            tokenizer (TokenizerWrapper): The tokenizer object that is required to tokenize text data.
            sample_key (str): The key to access the sample in the BatchEncoding.
            index_path (Optional[Path], optional): The path to the index file which indicates
              the start character/byte position of documents. Defaults to None.
            jq_pattern (str, optional): The jq pattern to filter the data. Results are afterwards tokenized and packed.
              Defaults to ".text".

        Returns:
            None
        """
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)

        self.reader = LargeFileLinesReader(self.raw_data_path, index_path=index_path)
        self.jq_filter = jq.compile(jq_pattern)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.reader)

    def __getitem__(self, idx: int) -> BatchEncoding:
        """
        Retrieves the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            BatchEncoding: The tokenized representation of the item.

        Raises:
            IndexError: If the index is out of bounds.
        """
        self._check_if_inbounds(idx)
        return self.tokenizer.tokenize(text=self.jq_filter.input_text(self.reader[idx]).first())


class PackedMemMapDatasetBase(Dataset):
    """PackedMemMapDatasetBase class."""

    DATA_SECTION_LENGTH_IN_BYTES = EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES
    TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES = EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES
    HEADER_SIZE_IN_BYTES = EmbeddedStreamData.HEADER_SIZE_IN_BYTES
    np_dtype_of_tokens_on_disk_from_bytes = {
        1: np.dtype(np.uint8).newbyteorder("<"),
        2: np.dtype(np.uint16).newbyteorder("<"),
        4: np.dtype(np.uint32).newbyteorder("<"),
    }
    type_converter_for_torch = {1: np.uint8, 2: np.int32, 4: np.int64}

    def __init__(self, raw_data_path: Path, sample_key: str):
        """
        Initializes the PackedMemMapDatasetBase object.

        Args:
            raw_data_path (Path): Path to a packed binary file (*.pbin).
                Use `modalities data pack_encoded_data` to create one based on a JSONL-file.
            sample_key (str): The key to access the sample in the BatchEncoding.

        Raises:
            RuntimeError: If the token representation with the given size is not supported.

        Returns:
            None

        Note:
            TODO: sample_key should support multi-modal features using separately encoded inputs,
                  this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)
        self._embedded_stream_data = EmbeddedStreamData(raw_data_path)
        self._token_size_in_bytes = self._embedded_stream_data.token_size_in_bytes
        try:
            self._token_dtype_on_disk = self.np_dtype_of_tokens_on_disk_from_bytes[self._token_size_in_bytes]
            self._token_dtype_in_ram = self.type_converter_for_torch[self._token_size_in_bytes]
        except KeyError:
            raise RuntimeError(
                f"Encountered a required token representation with {self._token_size_in_bytes},"
                " which is not supported. Consider using a smaller vocabulary."
            )
        self._index = self._generate_packing_index()

    def _generate_packing_index(self) -> list[tuple[int, int]]:
        # Generates the packing index for the dataset.
        # The index is list of tuples, where each tuple contains the offset and length in bytes.

        return self._embedded_stream_data.index_base

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._index)

    def __getitem__(self, idx: int) -> BatchEncoding:
        """
        Retrieves the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            BatchEncoding: The retrieved item as a BatchEncoding object.

        Raises:
            ValueError: If the length of the sample in bytes is not a multiple of `self._token_size_in_bytes`.
        """
        self._check_if_inbounds(idx)
        # offset and length in bytes
        offset_in_bytes, length_in_bytes = self._index[idx]
        if length_in_bytes % self._token_size_in_bytes != 0:
            raise ValueError(
                f"Length of the sample in bytes is not a multiple of {self._token_size_in_bytes}."
                f"Offset in bytes: {offset_in_bytes}, Length in bytes: {length_in_bytes}"
            )
        # numpy frombuffer takes the memmap object as the buffer
        # and indices the data section with the given offset (in bytes)
        # and length in indices of type self._token_dtype_on_disk
        num_tokens = length_in_bytes // self._token_size_in_bytes
        tokens = np.frombuffer(
            buffer=self._embedded_stream_data.data,
            dtype=self._token_dtype_on_disk,
            count=num_tokens,
            offset=offset_in_bytes,
        )
        # torch can't convert most uint-formats, therefore we infer regular int types
        tokens = tokens.astype(self._token_dtype_in_ram)
        return BatchEncoding(data={self.sample_key: tokens})


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    """PackedMemMapDatasetContinuous class."""

    def __init__(self, raw_data_path: Path, sample_key: str, block_size: int):
        """
        Initializes the PackedMemMapDatasetContinuous object.

        Args:
            raw_data_path (Path): Path to a packed binary file (*.pbin).
                Use `modalities data pack_encoded_data` to create one based on a JSONL-file.
            sample_key (str): The key to access the sample in the BatchEncoding.
            block_size (int): The size of the block.

        Returns:
            None
        """
        self.block_size = block_size
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)

    def _generate_packing_index(self) -> list[tuple[int, int]]:
        # Generates the packing index for the dataset.
        # A list of tuples representing the index, where each tuple contains the offset and length in bytes.

        # get number of total tokens in file
        total_tokens = self._embedded_stream_data.data_len // self._token_size_in_bytes
        if total_tokens < self.block_size:
            raise ValueError(
                f"Block size ({self.block_size}) is larger than the"
                "total number of tokens in the dataset ({total_tokens})."
            )
        if self.block_size < 2:
            raise ValueError("Block size must be at least 2.")
        # Given a fixed number of samples we can compute the total number of tokens as
        # num_tokens = block_size + (block_size-1) * (num_samples-1)
        # as the first sample always needs block_size many tokens and the following samples
        # each need block_size-1 many tokens (since we can reuse the last target token as the first input token
        # of the subsequent sample).
        num_samples = (total_tokens - self.block_size) // (self.block_size - 1) + 1
        # given num_samples we calculate the starting index and length of each sample as tuple.
        return [
            ((i * self.block_size - i) * self._token_size_in_bytes, self.block_size * self._token_size_in_bytes)
            for i in range(num_samples)
        ]


class PackedMemMapDatasetMegatron(PackedMemMapDatasetBase):
    def __init__(self, raw_data_path: Path, sample_key: str, block_size: int):
        self.block_size = block_size
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)

    def _generate_packing_index(self) -> list[tuple[int, int]]:
        index = []
        curr_offset = self.HEADER_SIZE_IN_BYTES
        curr_len = 0
        block_size_in_bytes = self.block_size * self._token_size_in_bytes
        for segment_offset, segment_len in tqdm(self._embedded_stream_data.index_base):
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
