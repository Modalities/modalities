import math
import pickle
from pathlib import Path
from typing import Iterator, Optional


import numpy as np
from tqdm import tqdm


class EmbeddedStreamData:
    # amount of bytes to represent number of all tokens in dataset.
    # If the amount exceeds 2^(8*`header_size_in_bytes`), this requires adaptation.
    # Decided to keep this constant, since a size of 8 bytes requires more data than the internet currently provides
    DATA_SECTION_LENGTH_IN_BYTES = 8
    TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES = 4
    HEADER_SIZE_IN_BYTES = DATA_SECTION_LENGTH_IN_BYTES + TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES

    def __init__(self, data_path: Path, load_index: Optional[bool] = True):
        """
        Initializes an EmbeddedStreamData object.

        Args:
            data_path (Path): The path to the packed data file.
            load_index (bool, optional): Whether to load the index. Defaults to True.

        Raises:
            FileNotFoundError: If the packed data file is not found at the specified path.

        """
        self._data_path = data_path
        if not self._data_path.is_file():
            raise FileNotFoundError(
                f"Packed Data was not found at {self._data_path.absolute()}."
                f"Create on in advance by using `modalities data pack_encoded_data`."
            )

        with self._data_path.open("rb") as f:
            # get number of bytes in data section
            data_section_length_in_bytes = f.read(self.DATA_SECTION_LENGTH_IN_BYTES)
            self.data_len = int.from_bytes(data_section_length_in_bytes, byteorder="little")

            # get number of bytes for encoding a single token
            f.seek(self.DATA_SECTION_LENGTH_IN_BYTES)
            token_size_as_bytes = f.read(self.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES)
            self.token_size_in_bytes = int.from_bytes(token_size_as_bytes, byteorder="little", signed=False)

            # get index
            if load_index:
                f.seek(self.HEADER_SIZE_IN_BYTES + self.data_len)
                pkl_encoded_index = f.read()
                # contains the start offset and length of each segment
                # as byte positions in the data section
                self._index_base: list[tuple[int, int]] = pickle.loads(pkl_encoded_index)
            else:
                self._index_base = None

            # initialize memmapped data section
            self._data = np.memmap(self._data_path, mode="r", offset=self.HEADER_SIZE_IN_BYTES, shape=(self.data_len,))

    @property
    def index_base(self) -> list[tuple[int, int]]:
        if self._index_base is None:
            raise ValueError("Index was not loaded. Set `load_index=True` during initialization.")
        return self._index_base

    @property
    def data(self) -> np.ndarray:
        return self._data


def join_embedded_stream_data(stream_data: list[EmbeddedStreamData], target_file: Path, chunk_size: int = 2048):
    """
    Joins the embedded stream data into a single file.

    Args:
        stream_data (list[EmbeddedStreamData]): A list of EmbeddedStreamData objects representing the stream data.
        target_file (Path): The target file to write the joined data to.
        chunk_size (int, optional): The size of each data chunk. Defaults to 2048.

    Raises:
        FileExistsError: If the target file already exists.

    Returns:
        None
    """
    if target_file.exists():
        raise FileExistsError(f'Target File at "{target_file}" exists!')
    data_len = sum(d.data_len for d in stream_data)
    assert len({d.token_size_in_bytes for d in stream_data}) == 1, (
        "Found different token representation sizes. This could indicate the usage of different tokenizers. "
        "Not supported!"
    )
    token_size_in_bytes = stream_data[0].token_size_in_bytes

    num_data_chunks = sum(math.ceil(d.data_len / chunk_size) for d in stream_data)
    data_stream_generator = (d.data[i : i + chunk_size] for d in stream_data for i in range(0, d.data_len, chunk_size))

    num_entries = sum(len(d.index_base) for d in stream_data)

    def index_stream_generator() -> Iterator[tuple[int, int]]:
        # generates a stream of index offsets and segment lengths.
        curr_offset = 0
        for embedded_stream_data in stream_data:
            for entry_offset, segment_length in embedded_stream_data.index_base:
                yield entry_offset + curr_offset, segment_length
            curr_offset += embedded_stream_data.data_len
            curr_offset -= embedded_stream_data.HEADER_SIZE_IN_BYTES

    with target_file.open("wb") as fout:
        fout.write(data_len.to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"))
        fout.write(
            token_size_in_bytes.to_bytes(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little")
        )
        for data_chunk in tqdm(data_stream_generator, total=num_data_chunks, desc="Writing Data Chunks..."):
            fout.write(data_chunk)

        joint_index = [entry for entry in tqdm(index_stream_generator(), total=num_entries, desc="Concatenating Index")]
        pickled_index = pickle.dumps(joint_index)
        pickled_index_as_chunks = (pickled_index[i : i + chunk_size] for i in range(0, len(pickled_index), chunk_size))
        num_index_chunks = math.ceil(len(pickled_index) / chunk_size)
        for index_chunk in tqdm(pickled_index_as_chunks, total=num_index_chunks, desc="Writing Index Chunks..."):
            fout.write(index_chunk)
