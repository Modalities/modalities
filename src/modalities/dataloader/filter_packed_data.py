import pickle
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from modalities.dataloader.create_packed_data import EmbeddedStreamData, update_data_length_in_pre_allocated_header
from modalities.dataloader.dataset import PackedMemMapDatasetBase


def filter_dataset(
    src_path: Path,
    dst_path: Path,
    filter_func: Callable[[tuple[int, dict[str, NDArray[np.int_]]]], bool],
) -> None:
    """
    Filters the dataset based on a given filter function and writes the filtered data to the destination path.
    Args:
        src_path (Path): The path to the source dataset to filter.
        dst_path (Path): The path where the filtered dataset will be written.
        filter_func (Callable[[tuple[int, dict[str, NDArray[np.int_]]]], bool]):
            A function that takes a sample index and its content and returns
            True if the sample should be included, False otherwise.
    Returns:
        None
    """
    sample_key: str = "input_ids"
    index_list: list[tuple[int, int]] = []
    source_data = PackedMemMapDatasetBase(src_path, sample_key=sample_key, load_index=True)
    with dst_path.open("wb") as f_out:
        # allocate first self.header_size_in_bytes bytes for header (encodes length of data section)
        # not possible to prepend header after determining size of data section
        f_out.write((0).to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"))
        tok_size = source_data.token_size_in_bytes
        tok_type = np.dtype(f"uint{tok_size * 8}")
        f_out.write(tok_size.to_bytes(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little"))
        # The offset only applies to the data section, not the header
        # When we load the file, we add the header size to the offset
        curr_offset = 0

        # Provide sample and its index (via enumerate) to the filter function.
        for _, entry in filter(filter_func, enumerate(tqdm(source_data, desc="Filtering samples"))):
            tokens: NDArray[np.int_] = entry[sample_key].astype(tok_type)
            tokens = tokens.astype(tokens.dtype.newbyteorder("<"))
            tokens_as_bytes = tokens.tobytes()
            f_out.write(tokens_as_bytes)
            segment_length = len(tokens_as_bytes)
            index_list.append((curr_offset, segment_length))
            curr_offset += segment_length
        # Write index at end of the file.
        f_out.write(pickle.dumps(index_list))

    update_data_length_in_pre_allocated_header(dst_path, index_list)
