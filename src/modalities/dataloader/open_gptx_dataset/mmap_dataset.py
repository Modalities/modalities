# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/mmap_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

import logging
import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
from numpy._typing import NDArray
from torch.utils.data import Dataset

from modalities.util import print_rank_0


def get_best_fitting_dtype(vocab_size: Optional[int] = None) -> np.dtype:
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def make_builder(out_file: str, dtype: Type):
    """Creates a dataset builder."""
    return MMapIndexedDatasetBuilder(out_file=out_file, dtype=dtype)


def bin_file_path_and_data_file_path_exists(path: str) -> bool:
    "Checks, whether path exists"
    return os.path.exists(get_index_file_path(path)) and os.path.exists(get_data_file_path(path))


def make_dataset(path: Union[str, Path], skip_warmup=False) -> Dataset:
    """Creates MMapIndexedDataset.
    :param path: path to .idx and .bin files
    :type path:
    :param skip_warmup:
    :type skip_warmup:
    :return:
    :rtype:
    """
    if isinstance(path, str):
        path = Path(path)
    logging.info(f"Loading mmap indexed dataset from {path}")
    if not bin_file_path_and_data_file_path_exists(path.__str__()):
        raise FileExistsError(
            f"mmap indexed dataset {get_index_file_path(path.__str__())} "
            f"or {get_data_file_path(path.__str__())} does not exist!"
        )
    return MMapIndexedDataset(path.__str__(), skip_warmup)


key_to_dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.single,
    7: np.double,
    8: np.uint16,
}


def code(dtype: Type) -> int:
    for k in key_to_dtypes.keys():
        if key_to_dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def get_index_file_path(prefix_path: str) -> str:
    return f"{prefix_path}.idx"


def get_data_file_path(prefix_path: str) -> str:
    return f"{prefix_path}.bin"


def _warmup_mmap_file(path: str):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


def exscan_from_cumsum_(arr):
    # given an array holding the result of an inclusive scan (cumsum),
    # convert to an exclusive scan (shift to the right)
    # [10, 30, 35, 50] --> [0, 10, 30, 35]
    if arr.size > 1:
        arr[1:] = arr[:-1]
    if arr.size > 0:
        arr[0] = 0


def get_pointers_with_total(sizes, elem_size, dtype) -> Tuple[np.ndarray, int]:
    """Return a numpy array of type np.dtype giving the byte offsets.

    Multiplies values in the sizes array by elemsize (bytes),
    and then computes an exclusive scan to get byte offsets.
    Returns the total number of bytes as second item in a tuple.
    """

    # scale values in sizes array by elemsize to get sizes in bytes
    pointers = np.array(sizes, dtype=dtype)
    pointers *= elem_size
    np.cumsum(pointers, axis=0, out=pointers)

    # get total number of bytes from all sizes (last element)
    bytes_last = pointers[-1] if len(sizes) > 0 else 0

    # convert to byte offsets. Shift sequence by 1 to the right and add 0 at first position
    exscan_from_cumsum_(pointers)

    return pointers, bytes_last


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        # Signature is used to check, whether binary file has been created based on this implementation
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @staticmethod
        def write_header(fout, dtype: Type, num_sizes: int, num_docs: int) -> int:
            """Writes header for mmap indexed dataset to
            given file handle, return number of bytes written."""

            # Get position of file handle
            start_pos = fout.tell()

            fout.write(MMapIndexedDataset.Index._HDR_MAGIC)
            # Indicates the version (8 bytes)
            fout.write(struct.pack("<Q", 1))
            # Code of data type used for saving tokens (1 byte)
            fout.write(struct.pack("<B", code(dtype)))
            # Number of documents (8 bytes)
            fout.write(struct.pack("<Q", num_sizes))
            # Number of documents + 1 (8 bytes)
            fout.write(struct.pack("<Q", num_docs))

            end_pos = fout.tell()

            return end_pos - start_pos

        @classmethod
        def writer(cls, path: str, dtype: Type):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")
                    return self

                @staticmethod
                def _get_pointers(sizes: np.ndarray, np_dtype):
                    """Return a numpy array of byte offsets given a list of sizes.

                    Multiplies values in the sizes array by dtype size (bytes),
                    and then computes an exclusive scan to get byte offsets.
                    """

                    # Compute element sizes in bytes
                    # sizes: [3, 4, 2] -> pointers: [0, 3, 7, 9]
                    pointers, _ = get_pointers_with_total(sizes, dtype().itemsize, np_dtype)
                    return pointers

                def write(self, sizes: List, doc_idx: List) -> None:
                    MMapIndexedDataset.Index.write_header(self._file, dtype, len(sizes), len(doc_idx))

                    # Write sizes to index
                    sizes32 = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes32.tobytes(order="C"))
                    del sizes32

                    # Write pointers to index
                    pointers = self._get_pointers(sizes, np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    # Write doc_idx to index
                    # example of doc_idx: [0, 1, 2, 3]. Note that "0" has been inserted
                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    assert isinstance(doc_idx, np.ndarray)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path: str, skip_warmup: bool = False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. " "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = key_to_dtypes[dtype_code]
                # Get number of bytes for representing data type
                self._dtype_size = self._dtype().itemsize

                # Corresponds to len(num_sizes)
                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            # "np.memmap" creates a numpy memory view, which is an array-like but not a proper "np.ndarray"
            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            # With "memoryview" the "np.memmap" view is converted into a python buffer
            # which can be read by "np.frombuffer", to get a proper "np.ndarray"
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        # TODO: check, why setting skip_warmup to True per default
        self._do_init(state, skip_warmup=True)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(get_index_file_path(self._path), skip_warmup)
        logging.info(f"Loaded  {get_index_file_path(self._path)}")

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(get_data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(get_data_file_path(self._path), mode="r", order="C")
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None) -> NDArray:
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        return np_array

    @property
    def sizes(self) -> List:
        return self._index.sizes

    def size(self, index) -> int:
        return self._index.sizes[index]

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self) -> List:
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_) -> None:
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self) -> bool:
        return False

    @staticmethod
    def exists(path) -> bool:
        return os.path.exists(get_index_file_path(path)) and os.path.exists(get_data_file_path(path))

    @property
    def dtype(self) -> Type:
        return self._index.dtype


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype: Type = np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes: List = []  # number of tokens per line/document
        # Contains the document offsets, e.g., [0,3,5] means that document 1 consists of the first 3 lines
        # and document 2 of the lines 3 to 5
        # i.e, maps the document id to the starting line of the respective document
        self._doc_idx = [0]

    def add_item(self, tensor: torch.Tensor) -> None:
        # Convert to numpy array
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        # Write bytes to file
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self) -> None:
        self._doc_idx.append(len(self._sizes))  # len(_sizes) equals number of lines/documents

    def merge_file_(self, file_to_merge: str) -> None:
        # Concatenate index
        index = MMapIndexedDataset.Index(get_index_file_path(file_to_merge))
        assert index.dtype == self._dtype

        total_len = len(index.sizes) + len(self._sizes)
        print(
            f"""\tconcat {file_to_merge} size={len(index.sizes)}
            for a total size of {total_len}"""
        )

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])

        # Concatenate data
        with open(get_data_file_path(file_to_merge), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file) -> None:
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
