import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import numpy as np
import torch.distributed

from ..util import dist_setup_info
from .create_index import IndexGenerator


class BaseReader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: Union[int, slice]) -> Union[str, List[str]]:
        raise NotImplementedError


# TODO: benchmark tokenized version vs plain text version (regarding speed and storage consumption)
class LargeFileLinesReader(BaseReader):
    def __init__(
        self,
        raw_data_path: Union[str, Path],
        index_path: Union[str, Path] = None,
        lazy_init: bool = False,
        synced_init: bool = True,
    ):
        """

        :param raw_data_path: Path a jsonl file, which holds text data
        :param index_path: Path to an index file, which is supposed to indicate the start character position
                           and length of samples given in `raw_data_path`.
                           If not defined, an index next to `raw_data_path` is picked,
                           by replacing its suffix with ".idx".
        :param lazy_init: In case no existing index file is found, it is generated on the fly if this option is toggled
        :param synced_init: Needs to get toggled if torch.distributed-processes access this objects initialization.
                            It uses a `torch.distributed.barrier` to sync these processes.
                            If there are multiple processes running, but only one actually reaches this part of code,
                            this must be turned off!
        """
        self.raw_data_path = Path(raw_data_path)
        self.index_path = self._default_index_path(index_path)

        if not self.raw_data_path.is_file():
            raise FileNotFoundError("Raw data file does not exist")
        if not lazy_init and not self.index_path.is_file():
            raise FileNotFoundError("Index file must exist when lazy init is turned off")

        if lazy_init:
            if synced_init:
                self.synced_index_initialization()
            else:
                self.lazy_index_initialization()

        with self.index_path.open("rb") as f:
            self.index = pickle.load(f)

    def synced_index_initialization(self):
        if dist_setup_info.rank == 0:
            self.lazy_index_initialization()
        else:
            print(f"Waiting for Rank 0 to initialize index. My Rank is {dist_setup_info.rank}")
        if dist_setup_info.dist_launched:
            torch.distributed.barrier()  # index creation is not threadsafe among ddp-processes

    def lazy_index_initialization(self):
        if not self.index_path.is_file():
            print("No Index File provided. Will generate one...")
            generator = IndexGenerator(self.raw_data_path)
            generator.run(self.index_path)

    def _default_index_path(self, index_path: Union[str, Path, None]) -> Path:
        if index_path is None:
            default_index_path = Path(self.raw_data_path.parent, f"{self.raw_data_path.stem}.idx")
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
