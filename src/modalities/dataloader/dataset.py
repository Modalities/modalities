from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Optional

import jq
import numpy as np
from pydantic import BaseModel
from torch.utils.data.dataset import Dataset as TorchdataSet
from tqdm import tqdm
from transformers import BatchEncoding

from modalities.dataloader.create_packed_data import EmbeddedStreamData
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


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
                raise NotImplementedError(f"DummyDataset does not support type {s.sample_type}")
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
        if idx >= len(self.reader):
            raise IndexError("Index out of bounds")
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

    def __init__(self, raw_data_path: Path, sample_key: str, load_index: Optional[bool] = True):
        """
        Initializes the PackedMemMapDatasetBase object.

        Args:
            raw_data_path (Path): Path to a packed binary file (*.pbin).
                Use `modalities data pack_encoded_data` to create one based on a JSONL-file.
            sample_key (str): The key to access the sample in the BatchEncoding.
            load_index (bool, optional): Flag indicating whether to load the index. Defaults to True.

        Raises:
            RuntimeError: If the token representation with the given size is not supported.

        Returns:
            None

        Note:
            TODO: sample_key should support multi-modal features using separately encoded inputs,
                  this needs to get replaced with a list of sample keys!
        """
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key)
        self._embedded_stream_data = EmbeddedStreamData(raw_data_path, load_index=load_index)
        self._token_size_in_bytes = self._embedded_stream_data.token_size_in_bytes
        try:
            self._token_dtype_on_disk = self.np_dtype_of_tokens_on_disk_from_bytes[self._token_size_in_bytes]
            self._token_dtype_in_ram = self.type_converter_for_torch[self._token_size_in_bytes]
        except KeyError as e:
            raise RuntimeError(
                f"Encountered a required token representation with {self._token_size_in_bytes},"
                " which is not supported. Consider using a smaller vocabulary."
            ) from e
        self._index = self._generate_packing_index()

    @property
    def token_size_in_bytes(self) -> int:
        return self._token_size_in_bytes

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

    def __getitem__(self, idx: int | slice) -> BatchEncoding:
        """
        Retrieves the item at the given index or a slice of items.

        Args:
            idx (int | sclice): The index of the item to retrieve or slice of items.

        Returns:
            BatchEncoding: The retrieved item as a BatchEncoding object.

        Raises:
            ValueError: If the length of the sample in bytes is not a multiple of `self._token_size_in_bytes`.
        """

        if not isinstance(idx, slice):
            # (offset_in_bytes, length_in_bytes)
            item_positions: list[tuple[int, int]] = [self._index[idx]]
        else:
            if idx.step is not None and idx.step != 1:
                raise ValueError("Slicing with step != 1 is not supported.")
            item_positions = self._index[idx]

        if len(item_positions) == 0:
            return BatchEncoding(data={self.sample_key: []})

        # numpy frombuffer takes the memmap object as the buffer
        # and indices the data section with the given offset (in bytes)
        # and length in indices of type self._token_dtype_on_disk
        num_bytes_stop = item_positions[-1][0] + item_positions[-1][1]
        num_bytes_start = item_positions[0][0]
        length_in_bytes = num_bytes_stop - num_bytes_start
        num_tokens = length_in_bytes // self._token_size_in_bytes
        tokens = np.frombuffer(
            buffer=self._embedded_stream_data.data,
            dtype=self._token_dtype_on_disk,
            count=num_tokens,
            offset=num_bytes_start,
        )
        # torch can't convert most uint-formats, therefore we infer regular int types
        tokens = tokens.astype(self._token_dtype_in_ram)

        documents = []
        for offset_in_bytes, length_in_bytes in item_positions:
            token_start = (offset_in_bytes - num_bytes_start) // self._token_size_in_bytes
            token_end = (offset_in_bytes + length_in_bytes - num_bytes_start) // self._token_size_in_bytes
            documents.append(tokens[token_start:token_end])

        # TODO: the return type is inconsistent here.
        # If idx is an integer, we return a BatchEncoding with a single document.
        # If idx is a slice, we return a BatchEncoding with a list of documents.
        if not isinstance(idx, slice):
            return BatchEncoding(data={self.sample_key: documents[0]})
        else:
            return BatchEncoding(data={self.sample_key: documents})


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    """PackedMemMapDatasetContinuous class."""

    def __init__(
        self,
        raw_data_path: Path,
        sample_key: str,
        block_size: int,
        reuse_last_target: bool,
        load_index: Optional[bool] = False,
    ):
        """
        Initializes a Dataset object for continuous packed data. If `reuse_last_target` is True,
        the last target token of one sample is reused as the first input token of the next sample,
        creating an overlap of one token between samples (recommended for pre-training).
        If `reuse_last_target` is False, there is no overlap:
        Each sample is a distinct block, and the first token of each sample is never used as a target
        (recommended for instruction tuning).

        Args:
            raw_data_path (Path): Path to a packed binary file (*.pbin).
                Use `modalities data pack_encoded_data` to create one based on a JSONL-file.
            sample_key (str): The key to access the sample in the BatchEncoding.
            block_size (int): The size of the block.
            load_index (bool, optional): Flag indicating whether to load the index.
                This is only needed for debugging purposes to index the original documents.
                The continuous packing does not need to load the index and should be
                deactivated as it significantly increases the instantiation time. Defaults to False.
            reuse_last_target (bool, optional): Whether to reuse the last target token between samples. Default: True.
                In pre-training, setting this to True enables overlapping samples:
                The last target token of one sample becomes the first input token of the next sample,
                maximizing data throughput.
                For instruction tuning (IT), where sequences are often short and may end with special tokens
                (e.g., PAD or EOS), set this to False to avoid overlap:
                Each token is used only once as a target and never reused as the start of the next sample.

                False: No overlap; each sample is a distinct block of tokens (recommended for instruction tuning).
                True: Overlap by one token;
                    each token (except the very first and last) is both a sample and a target
                    (recommended for pre-training).
        Returns:
            None
        """
        self.block_size = block_size
        self.reuse_last_target = reuse_last_target
        # TODO passing the load_index flag does not really comply with the inversion
        # of control principle. We should refactor this in the future.
        super().__init__(raw_data_path=raw_data_path, sample_key=sample_key, load_index=load_index)

    @staticmethod
    def _create_packed_index(
        total_tokens: int, block_size: int, token_size_in_bytes: int, reuse_last_target: bool
    ) -> np.ndarray:
        if reuse_last_target:
            # Given a fixed number of samples we can compute the total number of tokens as
            # num_tokens = block_size + (block_size-1) * (num_samples-1)
            # as the first sample always needs block_size many tokens and the following samples
            # each need block_size-1 many tokens (since we can reuse the last target token as the first input token
            # of the subsequent sample).
            num_samples = (total_tokens - block_size) // (block_size - 1) + 1
            i_array = np.arange(num_samples)
            sample_start_positions_bytes = (i_array * block_size - i_array) * token_size_in_bytes
        else:
            # If not reusing the last token, each sample consists of exactly block_size tokens
            # with no overlap between samples
            num_samples = total_tokens // block_size
            i_array = np.arange(num_samples)
            sample_start_positions_bytes = (i_array * block_size) * token_size_in_bytes
        sample_length_bytes = np.full(num_samples, block_size * token_size_in_bytes)
        result = np.stack((sample_start_positions_bytes, sample_length_bytes), axis=1)
        return result

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

        result = self._create_packed_index(
            total_tokens, self.block_size, self._token_size_in_bytes, self.reuse_last_target
        )
        return result


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


class CombinedDataset(Dataset):
    """Combines multiple datasets into one large dataset at runtime.

    Note: When using this class to combine multiple `PackedMemMapDataset`s, each packed sample
    is packed from a single dataset (i.e., the samples are not mixed between datasets).
    In the Dataloader, a batch will still contain packed samples from different datasets.
    """

    def __init__(self, datasets: list[Dataset]):
        """Initializes the CombinedDataset object, combining multiple datasets.

        Args:
            datasets (list[Dataset]): A list of datasets to combine.
        """
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in datasets], dtype=np.int64)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> dict:
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
        local_idx = idx - (self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0)

        return self.datasets[dataset_idx][local_idx]


class WeightedCombinedDataset(Dataset):
    """Combines multiple datasets at runtime, each contributing a chosen number of epochs.

    `CombinedDataset` concatenates its datasets once each, so the only way to change a
    dataset's share of the blend is to change how much of it is on disk. This class
    takes a repeat factor per dataset instead: 2.5 draws a dataset two and a half times
    per epoch, 0.3 draws three tenths of it. Nothing is duplicated on disk, fractional
    factors work, and the blend becomes a config value.

    The fractional part is realised without storing an index map, so memory stays
    constant no matter how large the datasets are. The documents making up a partial
    pass are picked by a seeded affine permutation of the dataset's indices, which
    spreads them evenly across the whole dataset rather than taking a prefix, and gives
    the same selection on every rank and every restart.

    Note:
        The partial-pass selection is evenly spread rather than statistically random.
        That is what makes it O(1), and it is a good property here -- a prefix would
        over-sample whatever the corpus happens to be ordered by -- but it is not a
        substitute for shuffling, which the sampler still does.
    """

    def __init__(self, datasets: list[Dataset], repeat_factors: list[float], seed: int = 42):
        """Initializes the WeightedCombinedDataset.

        Args:
            datasets (list[Dataset]): The datasets to combine.
            repeat_factors (list[float]): How many times to draw each dataset per
                epoch. Must be non-negative and align one-to-one with `datasets`.
                A factor of 0 excludes a dataset while keeping it declared.
            seed (int): Seed for the partial-pass selection.

        Raises:
            ValueError: If the lengths disagree or a factor is negative.
        """
        if len(datasets) != len(repeat_factors):
            raise ValueError(
                f"got {len(datasets)} datasets but {len(repeat_factors)} repeat factors; they must correspond"
            )
        if any(factor < 0 for factor in repeat_factors):
            raise ValueError(f"repeat factors must be non-negative, got {repeat_factors}")

        self.datasets = datasets
        self.repeat_factors = list(repeat_factors)
        self.seed = seed

        self._full_passes: list[int] = []
        self._num_partial: list[int] = []
        self._permutation_params: list[tuple[int, int]] = []
        virtual_lengths: list[int] = []

        for dataset_idx, (dataset, factor) in enumerate(zip(datasets, repeat_factors)):
            num_samples = len(dataset)
            full_passes = int(factor)
            num_partial = int(round((factor - full_passes) * num_samples))
            # Rounding up to a whole extra pass is expressed as one more full pass, so
            # `num_partial` never equals `num_samples` and the permutation stays a
            # strict subset.
            if num_partial >= num_samples > 0:
                full_passes += 1
                num_partial = 0
            self._full_passes.append(full_passes)
            self._num_partial.append(num_partial)
            self._permutation_params.append(self._affine_permutation_params(num_samples, seed, dataset_idx))
            virtual_lengths.append(full_passes * num_samples + num_partial)

        self.cumulative_sizes = np.cumsum(virtual_lengths, dtype=np.int64)

    @staticmethod
    def _affine_permutation_params(num_samples: int, seed: int, dataset_idx: int) -> tuple[int, int]:
        # `index -> (multiplier * index + offset) % num_samples` is a bijection exactly
        # when the multiplier is coprime with num_samples, which is what makes the
        # partial pass a subset with no repeats.
        if num_samples <= 1:
            return 1, 0
        rng = np.random.default_rng([seed, dataset_idx])
        multiplier = 1
        for _ in range(1000):
            candidate = int(rng.integers(1, num_samples))
            if math.gcd(candidate, num_samples) == 1:
                multiplier = candidate
                break
        return multiplier, int(rng.integers(0, num_samples))

    def __len__(self) -> int:
        """Returns the number of samples one epoch of the blend yields.

        Returns:
            int: Sum over datasets of `repeat_factor * len(dataset)`, rounded per
                dataset.
        """
        return int(self.cumulative_sizes[-1]) if len(self.cumulative_sizes) else 0

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a sample from the blend.

        Args:
            idx (int): Index into the blend.

        Returns:
            dict: The sample from whichever dataset the index falls in.

        Raises:
            IndexError: If `idx` is outside the blend.
        """
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"index {idx} is out of range for a blend of {len(self)} samples")

        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx, side="right"))
        local_idx = idx - (self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0)

        num_samples = len(self.datasets[dataset_idx])
        num_in_full_passes = self._full_passes[dataset_idx] * num_samples
        if local_idx < num_in_full_passes:
            sample_idx = local_idx % num_samples
        else:
            multiplier, offset = self._permutation_params[dataset_idx]
            sample_idx = (multiplier * (local_idx - num_in_full_passes) + offset) % num_samples

        return self.datasets[dataset_idx][int(sample_idx)]
