import pickle
from pathlib import Path
from typing import List, Union

import numpy as np

from .create_index import IndexGenerator


# TODO: benchmark tokenized version vs plain text version (regarding speed and storage consumption)
class LargeFileLinesReader:
    def __init__(
        self,
        raw_data_path: Union[str, Path],
        index_path: Union[str, Path] = None,
        lazy_init: bool = False,
        max_lines: int = None,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.index_path = self._default_index_path(index_path)
        self.max_lines = max_lines

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

    def _default_index_path(self, index_path: Union[str, Path, None]) -> Path:
        if index_path is None:
            default_index_path = Path(self.raw_data_path.parent, f"{self.raw_data_path.stem}.idx.pkl")
            print(f"No specific Index Path provided. Creating Index next to input data at: {default_index_path}")
            return default_index_path
        return Path(index_path)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, key: Union[int, slice]) -> Union[str, List[str]]:
        if isinstance(key, slice):
            return [self.__read_from_raw_file(*idx) for idx in self.index[key]]
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
