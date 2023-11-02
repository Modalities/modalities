import pickle
from pathlib import Path
from typing import Union

import numpy as np
from create_index import IndexGenerator


class LargeFileLinesReader:
    def __init__(
        self,
        raw_data_path: Union[str, Path],
        index_path: Union[str, Path],
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

    def __getitem__(self, key: int) -> str:
        if key >= len(self) or key < len(self):
            raise IndexError()
        offset, length_of_bytestream = self.index[key]
        return self.__read_from_raw_file(offset, length_of_bytestream)

    def __read_from_raw_file(self, offset: int, length_of_bytestream: int) -> str:
        def safe_decoder(byte_char):
            try:
                c = byte_char.decode("iso-8859-1")
            except Exception:
                c = ""
                print('Encountered invalid char: "{byte_char}"')
            return c

        string = (
            np.memmap(self.raw_data_path, mode="r", offset=offset, shape=(length_of_bytestream,)).view("S1").tolist()
        )
        decoded_string = []
        for c in string:
            decoded_string.append(safe_decoder(c))
        return "".join(decoded_string)


# TODO:: move to test
# # raw_path = Path("/home/shared/openwebtext/pile_openwebtext2_en.jsonl")
# raw_path = Path("/home/shared/openwebtext/head20000_openwebtext2_en.jsonl")
# # raw_path = Path("/home/haag/Documents/projects/mmap/a.jsonl")
# index_path = Path("map.pickle")
# # index_path = Path("/home/haag/Documents/projects/mmap/map.pickle")

# reader = LargeFileLinesReader(raw_path, index_path, lazy_init=True)
# print(len(reader))
# for i in range(3):
#     print(reader[i])
#     print(reader[-i])
#     print("-"*50)


# print(reader[213])
# print(reader[21])
# print(reader[2113])

# # from time import time
# # s_t = time()
# # for i in range(len(reader)):
# #     s = reader[i]
# # e_t = time()

# # print(f"throughput {len(reader)/(e_t-s_t)}s")
