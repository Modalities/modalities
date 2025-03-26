import math
import os
import pickle
from itertools import repeat
from pathlib import Path
from typing import BinaryIO

import numpy as np

from modalities.dataloader.create_packed_data import EmbeddedStreamData
from modalities.utils.logging import get_logger


class TokenizedFileWriter:
    @staticmethod
    def write_tokenized_dataset(
        tokenized_dataset: list[np.ndarray],
        tokenized_dataset_file_path: Path,
        write_batch_size: int = 10000,
        token_size_in_bytes: int = None,
    ) -> None:
        """Writes a tokenized dataset to disc in the custom pbin file format.

        Args:
            tokenized_dataset (list[np.ndarray]): The tokenized dataset to write to disc.
            tokenized_dataset_file_path (Path): The path to the tokenized dataset file.
            token_size_in_bytes (int): The number of bytes for a single token.
        """

        if token_size_in_bytes is None:
            get_logger().warning(
                "Token size in bytes not provided, calculating token size based on the maximum token in the dataset."
            )
            token_size_in_bytes = TokenizedFileWriter.get_required_num_of_bytes_to_repr(
                np.max([np.max(sample) for sample in tokenized_dataset])
            )

        with tokenized_dataset_file_path.open("wb") as chunk_file:
            TokenizedFileWriter._write_initial_header_segment(chunk_file, token_size_in_bytes)
            index_list = TokenizedFileWriter._write_data_segment(
                file_descriptor=chunk_file,
                token_data=tokenized_dataset,
                token_size_in_bytes=token_size_in_bytes,
                write_batch_size=write_batch_size,
            )
            TokenizedFileWriter._write_index_segment(chunk_file, index_list)
        if len(index_list) > 0:
            TokenizedFileWriter._update_data_length_in_initial_header(tokenized_dataset_file_path, index_list)
        else:
            # normally we could have checked this in the beginning via len(tokenized_dataset) == 0
            # but if the dataset is processed lazily via a generator, len(...) would load
            # the whole dataset into memory
            os.remove(tokenized_dataset_file_path)
            raise ValueError("The tokenized dataset did not create any data.")

    @staticmethod
    def _write_initial_header_segment(file_descriptor, token_size_in_bytes: int) -> None:
        # allocate first self.header_size_in_bytes bytes for header (encodes length of data section)
        # not possible to prepend header after determining size of data section
        file_descriptor.write((0).to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"))
        file_descriptor.write(
            token_size_in_bytes.to_bytes(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little")
        )

    @staticmethod
    def _update_data_length_in_initial_header(tokenized_dataset_file_path: Path, index_list: list[tuple[int, int]]):
        # Update the length of the data section in the pre-allocated header of the destination file.
        # The data segment length is sum of the starting position and the length of the last document.
        length_of_byte_encoded_data_section = index_list[-1][0] + index_list[-1][1]
        data_section_length_in_bytes = length_of_byte_encoded_data_section.to_bytes(
            EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"
        )
        with tokenized_dataset_file_path.open("rb+") as fout:
            fout.seek(0)
            fout.write(data_section_length_in_bytes)

    @staticmethod
    def _write_index_segment(file_descriptor: BinaryIO, index_list: list[tuple[int, int]]) -> None:
        file_descriptor.write(pickle.dumps(index_list))

    @staticmethod
    def _write_data_segment(
        file_descriptor: BinaryIO, token_data: list[np.ndarray], token_size_in_bytes: int, write_batch_size: int
    ) -> list[tuple[int, int]]:
        def encoded_token_to_bytes(encoded_token: int, token_size_in_bytes: int) -> bytes:
            # Converts an token_ids to its byte representation.
            try:
                token_bytes = encoded_token.to_bytes(token_size_in_bytes, byteorder="little", signed=False)
            except OverflowError as e:
                raise ValueError(f"Token {encoded_token} cannot be represented by {token_size_in_bytes} bytes.") from e
            return token_bytes

        samples = []
        index_list = []
        curr_offset = 0
        for sample_tokens in token_data:
            # convert token_ids to byte representation
            sample_token_byte_string = b"".join(
                map(encoded_token_to_bytes, sample_tokens.tolist(), repeat(token_size_in_bytes))
            )
            samples.append(sample_token_byte_string)
            index_list.append((curr_offset, len(sample_token_byte_string)))
            curr_offset += len(sample_token_byte_string)
            if len(samples) % write_batch_size == 0:
                file_descriptor.write(b"".join(samples))
                samples = []
        if len(samples) > 0:
            file_descriptor.write(b"".join(samples))
        return index_list

    @staticmethod
    def get_required_num_of_bytes_to_repr(int_to_get_repr: int) -> int:
        """
        Calculates the required number of bytes to represent an integer.

        Args:
            int_to_get_repr (int): The integer to get the representation for.

        Returns:
            int: The number of bytes required to represent the integer.
        """
        # we currently only support token sizes of 1, 2 and 4 bytes, as implemented here:
        # https://github.com/Modalities/modalities/blob/fix_char_bytes_indexation_mismatch/src/modalities/dataloader/dataset.py#L202
        num_bytes = math.ceil(math.log2(int_to_get_repr) / 8)
        if num_bytes == 1:
            return 1
        elif num_bytes == 2:
            return 2
        elif num_bytes <= 4:
            return 4
        else:
            raise ValueError("Currently only support token byte sizes of 1, 2, and 4.")
