import pickle
from pathlib import Path
from typing import Union

import numpy as np

from .create_index import IndexGenerator


# TODO: benchmark tokenized version vs plain text version (regarding speed and storage consumption)
class LargeFileLinesReader:
    def __init__(
        self,
        raw_data_path: Union[str, Path],
        index_path: Union[str, Path],  # TODO: default pointer to source data path
        lazy_init: bool = False,
        max_lines: int = None,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.index_path = Path(index_path)
        self.max_lines = max_lines

        # do some error checking
        if not self.raw_data_path.is_file():
            raise FileNotFoundError("Raw data file does not exist")
        if not lazy_init and not self.index_path.is_file():
            raise FileNotFoundError("Index file must exist when lazy init is turned off")

        if lazy_init and not self.index_path.is_file():
            print("No Index File provided. Will generate one...")
            generator = IndexGenerator(self.raw_data_path)
            generator.run(self.index_path)

        with self.index_path.open("rb") as f:
            self.index = pickle.load(f)

    def __len__(self) -> int:
        return len(self.index)

    # TODO: implement handling of slices
    def __getitem__(self, key: int) -> str:
        if key >= len(self) or key < -len(self):
            raise IndexError()
        offset, length_of_bytestream = self.index[key]
        return self.__read_from_raw_file(offset, length_of_bytestream)

    def __read_from_raw_file(self, offset: int, length_of_bytestream: int) -> str:
        def safe_decoder(byte_char):
            try:
                c = byte_char.decode("iso-8859-1")
            except Exception:
                c = ""
                print(f'Encountered invalid char: "{byte_char}"')
            return c

        string = (
            np.memmap(self.raw_data_path, mode="r", offset=offset, shape=(length_of_bytestream,)).view("S1").tolist()
        )
        decoded_string = []
        for c in string:
            decoded_string.append(safe_decoder(c))
        return "".join(decoded_string)
