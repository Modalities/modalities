import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BaseReader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: int | slice) -> str | List[str]:
        raise NotImplementedError


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
            raise FileNotFoundError("Index file does not exist. Use `modalities data create_raw_index` to create one.")

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
        f = self.raw_data_path.open()
        f.seek(offset)
        return f.read(sample_length_in_bytes)
