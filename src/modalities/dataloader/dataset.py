from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jq
import numpy as np
import sentencepiece as spm
from pydantic import BaseModel
from torch.utils.data.dataset import Dataset as TorchdataSet
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer

from ..dataloader.large_file_lines_reader import LargeFileLinesReader
from .create_packed_data import EmbeddedStreamData


class TokenizerWrapper:
    def __call__(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: Optional[bool] = None,
        truncation: Optional[bool] = None,
    ):
        raise NotImplementedError("Tokenizer must be implemented by a subclass.")


class PreTrainedHFTokenizer(TokenizerWrapper):
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, text, max_length, padding, truncation):
        return self.tokenizer.__call__(text, max_length=max_length, padding=padding, truncation=truncation)


class PreTrainedSPTokenizer(TokenizerWrapper):
    def __init__(self, tokenizer: spm.SentencePieceProcessor = None):
        self.tokenizer = tokenizer
        self.tokenizer_config = None
        self.continuation_tokenizer = None
        self.add_prefix_space = None
        self.read_model_proto()

    def __call__(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: Optional[bool] = None,
        truncation: Optional[bool] = None,
    ):
        return self.tokenizer.encode(text)


class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: Path, block_size: int, sample_key: str):
        self.raw_data_path = raw_data_path
        self.block_size = block_size
        self.sample_key = sample_key

    def _check_if_inbounds(self, idx: int):
        if not 0 <= idx < len(self):
            raise IndexError


class DummySampleDataType(str, Enum):
    FLOAT = "float"
    INT = "int"


class DummySampleConfig(BaseModel):
    sample_key: str
    sample_shape: Tuple[int, ...]
    sample_type: DummySampleDataType


class DummyDatasetConfig(BaseModel):
    num_samples: int
    sample_definition: List[DummySampleConfig]


class DummyDataset(Dataset):
    def __init__(self, num_samples: int, sample_definition: Tuple[DummySampleConfig]):
        """
        :param num_samples: Number of samples the dataset should generate.
        :param sample_definition: A list of tuples defining the dataset output.
            Each touple contains the sample key, shape and data type.
        """
        super().__init__(raw_data_path=None, block_size=None, sample_key=None)
        self.num_samples = num_samples
        self.sample_definition = sample_definition

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        return self._create_random_sample()

    def _create_random_sample(self):
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
        block_size: int,
        tokenizer: TokenizerWrapper,
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
    DATA_SECTION_LENGTH_IN_BYTES = EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES
    TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES = EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES
    HEADER_SIZE_IN_BYTES = EmbeddedStreamData.HEADER_SIZE_IN_BYTES
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
                              Use `modalities data pack_encoded_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param sample_key: model-specific parameter to indicate where in the BatchEncoding the input_token_ids are.
                           TODO: If this setting should support multi-modal features using separately encoded inputs,
                            this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size, sample_key=sample_key)
        self._embedded_stream_data = EmbeddedStreamData(raw_data_path)
        self._token_size_in_bytes = self._embedded_stream_data.token_size_in_bytes
        try:
            self._token_dtype = self.np_dtype_from_num_bytes[self._token_size_in_bytes]
        except KeyError:
            raise RuntimeError(
                f"Encountered a required token representation with {self._token_size_in_bytes},"
                " which is not supported. Consider using a smaller vocabulary."
            )
        self._index = self._generate_packing_index()

    def _generate_packing_index(self) -> List[Tuple[int, int]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> BatchEncoding:
        self._check_if_inbounds(idx)
        offset, length = self._index[idx]
        tokens = np.frombuffer(self._embedded_stream_data.data, dtype=self._token_dtype, count=length, offset=offset)
        return BatchEncoding(data={self.sample_key: tokens})


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    def _generate_packing_index(self) -> List[Tuple[int, int]]:
        # get number of total tokens in file
        total_tokens = self._embedded_stream_data.data_len // self._token_size_in_bytes
        num_samples = total_tokens // self.block_size
        return [(i * self.block_size * self._token_size_in_bytes, self.block_size) for i in range(num_samples)]


class PackedMemMapDatasetMegatron(PackedMemMapDatasetBase):
    def _generate_packing_index(self) -> List[Tuple[int, int]]:
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
