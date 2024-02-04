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

from .codecs import FixSizedCodec
from .create_packed_data import PackedDataGenerator
from .large_file_lines_reader import LargeFileLinesReader

class SampleKeysMismatchException(Exception):
    pass

class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: Path, sample_keys: list[str]):
        self.raw_data_path = raw_data_path
        self.sample_keys = sample_keys
        # must provide a sample key for each codec
        if len(self.sample_keys) != self.num_elements_per_item:
            raise SampleKeysMismatchException(
                "Expected %i sample keys, got %s" % (self.num_elements_per_item, self.sample_keys)
            )

    @property
    def num_elements_per_item(self) -> int:
        raise NotImplementedError

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
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)
        self.block_size = block_size

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
    
    def _read_bytes(self, offset: int, size: int) -> bytes:
        return np.memmap(
            self.raw_data_path,
            mode="r",
            offset=offset,
            shape=(size,),
        ).view(f"S{size}")[0]
    
    @property
    def num_elements_per_item(self) -> int:
        return len(self._codec_types)

    def __init__(self, raw_data_path: Path, sample_keys: list[str]):
        """
        Base class for packed memmapped datasets. The underlying dataset file has the structure:
        | data_header | codecs_header | data | codecs | index |

        The data and codecs headers contains information about the length of the data and codecs sequences.

        The codecs sequence contains the codec type hints required to decode the bytes to the expected
        data type. Specifically it is an encoded list of codec type names:

            (codec_1, codec_2, ...)

        The index stores byte positions of the dataset items in the following format:
        
            (offset, size_1, size_2, ...)

        The start and end tuple of the j-th value are computed by:

            (offset + sum_{i<j} size_i, offset + sum_{i<=j} size_i)

        Finally, the bytes can be decoded using the j-th codec type.

        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `modalities create_packed_data` to create one based on a jsonl-file.
        :param sample_keys: model-specific parameter to indicate where in the BatchEncoding the input fields are.
                            Specifically the j-th sample key provides the sample key to the j-th element
                            (i.e. size_j and codec_j).
        """

        self.raw_data_path = raw_data_path
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

        def read_header(offset: int, size: int) -> int:
            # read bytes from file
            return int.from_bytes(
                self._read_bytes(offset, size),
                byteorder="big"
            )

        # read headers
        self.data_len = read_header(
            offset=0,
            size=PackedDataGenerator.DATA_HEAD_SIZE_IN_BYTES
        )
        self.codecs_len = read_header(
            offset=PackedDataGenerator.DATA_HEAD_SIZE_IN_BYTES,
            size=PackedDataGenerator.CODECS_HEAD_SIZE_IN_BYTES
        )

        # compute offsets to index raw data file
        self.data_offset = (
            PackedDataGenerator.DATA_HEAD_SIZE_IN_BYTES +
            PackedDataGenerator.CODECS_HEAD_SIZE_IN_BYTES
        )
        self.codecs_offset = self.data_offset + self.data_len
        self.index_offset = self.codecs_offset + self.codecs_len

        # read codecs
        self._codec_type_hints = self._read_bytes(
            offset=self.codecs_offset,
            size=self.codecs_len
        )
        self._codec_type_hints = pickle.loads(self._codec_type_hints)
        # needs to be here to avoid circular import
        # TODO: find a better way to avoid the circular import
        from ..config.lookup_types import CodecTypes
        # resolve codec types
        self._codec_types = [
            getattr(CodecTypes, codec_type_hint).value
            for codec_type_hint in self._codec_type_hints
        ]

        # get index
        self._index_base = self._read_bytes(
            offset=self.index_offset,
            size=self.total_bytes - self.index_offset
        )
        self._index_base = pickle.loads(self._index_base)
        assert all(len(idx) == len(self._codec_types) + 1 for idx in self._index_base)
        
        # initialize after codec types are defined because
        # num_elements_per_item depends on it
        super().__init__(
            raw_data_path=raw_data_path, sample_keys=sample_keys
        )


class PackedMemMapDataset(PackedMemMapDatasetBase):
    """Packed Memory Map Dataset"""

    def __len__(self) -> int:
        return len(self._index_base)

    def __getitem__(self, idx: int) -> BatchEncoding:
        # get index values
        self._check_if_inbounds(idx)
        idx = self._index_base[idx]

        enc = {}
        offset = idx[0]
        for key, size, codec_type in zip(
            self.sample_keys, idx[1:], self._codec_types
        ):
            # decode item
            bytestring = self._read_bytes(offset, size)
            enc[key] = codec_type.decode(bytestring)
            # update offset
            offset += size

        return BatchEncoding(data=enc)


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    
    def __init__(self, raw_data_path: Path, sample_key: str, block_size: int):
        """
        PackedMemMapDatasetContinuous iterates through the data in block_size sized chunks,
        irrespective of the samples' start and end position, as defined in the index.
        Therefore, for this datset, the index is irrelevant.

        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `modalities create_packed_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param sample_key: model-specific parameter to indicate where in the BatchEncoding the input_token_ids are.
        """
        try:
            super().__init__(raw_data_path=raw_data_path, sample_keys=[sample_key])
        except SampleKeysMismatchException as e:
            raise ValueError(
                "Can only read continuously from packed data files of single-element dataset, i.e."
                "datasets with a single item per line. The specified dataset has %i elements per item."
                % self.num_elements_per_item
            ) from e

        # check if codec is supported
        if not issubclass(self.codec_type, FixSizedCodec):
            raise ValueError(
                "Can only read continuously from fix-sized codecs, got %s."
                % self.codec_type
            )

        self.block_size = block_size
        # get number of total tokens in file
        total_values = self.data_len // self._num_bytes_per_value
        self._num_samples = total_values // self.block_size
    
    @property
    def sample_key(self) -> str:
        return self.sample_keys[0]

    @property
    def codec_type(self) -> FixSizedCodec:
        return self._codec_types[0]

    @property
    def _num_bytes_per_value(self) -> int:
        return self.codec_type.num_bytes_per_value()

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> BatchEncoding:
        self._check_if_inbounds(idx)
        # read block-sized chunk of bytes
        byte_string = self._read_bytes(
            offset=self.data_offset + idx * self.block_size * self._num_bytes_per_value,
            size=self.block_size * self._num_bytes_per_value
        )
        # decode and pack into batch encoding
        values = self.codec_type.decode(byte_string)
        return BatchEncoding(data={self.sample_key: values})


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
        self._check_if_inbounds(idx)
        offset, length = self._index[idx]
        tokens_as_byte_strings = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=offset,
            shape=(length,),
        ).view(f"S{self.INT_SIZE_IN_BYTES}")
        tokens = [int.from_bytes(token, byteorder="big") for token in tokens_as_byte_strings]
        return BatchEncoding(data={self.sample_key: tokens})
