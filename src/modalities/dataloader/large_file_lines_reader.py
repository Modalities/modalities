import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class BaseReader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: int | slice) -> str | List[str]:
        raise NotImplementedError


# TODO: benchmark tokenized version vs plain text version (regarding speed and storage consumption)
class LargeFileLinesReader(BaseReader):
    def __init__(self, raw_data_path: Path, index_path: Path = None):
        """
        :param raw_data_path: Path to a jsonl file, which holds text data
        :param index_path: Path to an index file, which indicates the start character/byte position
                           and length of samples given in `raw_data_path`.
                           If not defined, an index next to `raw_data_path` is picked,
                           by replacing its suffix with ".idx".
        """
        self.raw_data_path = raw_data_path
        self.index_path = self.default_index_path(self.raw_data_path, index_path)

        if not self.raw_data_path.is_file():
            raise FileNotFoundError("Raw data file does not exist")
        if not self.index_path.is_file():
            raise FileNotFoundError("Index file does not exist. Use `modalities create_memmap_index` to create one.")

        with self.index_path.open("rb") as f:
            self.index = pickle.load(f)

    @staticmethod
    def default_index_path(raw_data_path: Path, index_path: Path = None) -> Path:
        if index_path is None:
            default_index_path = Path(raw_data_path.parent, f"{raw_data_path.stem}.idx")
            print(f"No specific Index Path provided. Pointing to index next to input data at: {default_index_path}")
            return default_index_path
        return index_path

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, key: int | slice) -> str | List[str]:
        if isinstance(key, slice):
            return [self.__read_from_raw_file(*idx) for idx in self.index[key]]
        offset, sample_length_in_bytes = self.index[key]
        return self.__read_from_raw_file(offset, sample_length_in_bytes)

    def __read_from_raw_file(self, offset: int, sample_length_in_bytes: int) -> str:
        def safe_decoder(byte_char):
            try:
                # TODO: verify why iso-8859-1 was necessary here in the path.
                #   Maybe there was an issue with the actual loading of the jsonl-files
                c = byte_char.decode("utf8")
            except Exception as exception:
                c = ""
                warnings.warn(f'Encountered invalid char: "{byte_char}".')
                warnings.warn(f"Encountered problem: {exception}")
            return c

        string = (
            np.memmap(self.raw_data_path, mode="r", offset=offset, shape=(sample_length_in_bytes,)).view("S1").tolist()
        )
        decoded_string = []
        for c in string:
            decoded_string.append(safe_decoder(c))
        return "".join(decoded_string)
