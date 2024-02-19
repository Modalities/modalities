import pickle
import warnings
from pathlib import Path
from typing import IO, Dict

import jq
import numpy as np
from tqdm import tqdm

from modalities.dataloader.codecs import Codec
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader


class PackedDataGenerator:
    """

    Format: HEAD DATA CODECS INDEX
    HEAD: DATA_HEAD CODECS_HEAD
    """

    # amount of bytes to represent number of all tokens in dataset.
    # If the amount exceeds 2^(8*`header_size_in_bytes`), this requires adaptation.
    # Decided to keep this constant, since a size of 8 bytes requires more data than the internet currently provides
    DATA_HEAD_SIZE_IN_BYTES = 8
    CODECS_HEAD_SIZE_IN_BYTES = 8

    def __init__(self, codecs: Dict[str, Codec], src_path: Path, idx_path: Path = None, max_num_of_bytes: int = None):
        """
        Reads in a jsonl file and the corresponding index file and packs dataset file for LLM training.
        :param codec: Codec object, which is used to encode the objects into bytes
        :param src_path: Path to a jsonl file, which holds text data
        :param index_path: Path to an index file, which indicates the start character position
                           and length of samples given in `src_path`.
                           If not defined, an index file next to `src_path` is picked,
                           by replacing its suffix with ".idx".
        :param jq_pattern: jq-pattern applied on every jsonl-entry. Results are afterwards tokenized and packed
        """

        jq_patterns, codecs = zip(*codecs.items())

        self.codecs = codecs
        self.jq_filters = [jq.compile(pattern) for pattern in jq_patterns]

        self.src_path = src_path
        self._reader = LargeFileLinesReader(src_path, index_path=idx_path)

        # keep track of file size
        self._total_data_bytes = 0
        self._max_data_bytes = max_num_of_bytes

        self._index_list = []

    @property
    def _current_offset(self) -> int:
        return self._total_data_bytes + type(self).DATA_HEAD_SIZE_IN_BYTES + type(self).CODECS_HEAD_SIZE_IN_BYTES

    def _default_destination_path(self, destination_path: Path = None) -> Path:
        if destination_path is None:
            default_destination_path = Path(self.src_path.parent, f"{self.src_path.stem}.pbin")
            print(
                f"No specific Destination Path provided. "
                f"Pointing to destination next to input data at: {default_destination_path}"
            )
            return default_destination_path
        return Path(destination_path)

    def run(self, dst_path: Path = None):
        assert self._total_data_bytes == 0, f"This {self.__name__} was already used and is exhausted. Use another!"
        dst_path = self._default_destination_path(destination_path=dst_path)

        if dst_path.exists():
            raise ValueError(f"file already exists at destination path '{dst_path}'.")

        with dst_path.open("wb") as f:
            # store the type-hints to the codec types
            # TODO: get the type hints from the enum in case they
            #       don't match the class name exactly
            codecs_bytes = pickle.dumps([type(codec).__name__ for codec in self.codecs])

            # allocate bytes for data header and write codecs header
            f.write((0).to_bytes(type(self).DATA_HEAD_SIZE_IN_BYTES, byteorder="big"))
            f.write(len(codecs_bytes).to_bytes(type(self).DATA_HEAD_SIZE_IN_BYTES, byteorder="big"))

            # write data section
            for idx, line in tqdm(enumerate(self._reader)):
                try:
                    self._process_line(f, line)
                except ValueError:
                    warnings.warn(f"Encountered empty sample in line {idx} of file {self.src_path}")
                except StopIteration:
                    break
                except Exception as exception:
                    warnings.warn(f"could not process line: {exception=}")

            # write codecs and index section to file
            f.write(codecs_bytes)
            f.write(pickle.dumps(self._index_list))

        self._update_data_length_in_pre_allocated_header(dst_path)

    def _update_data_length_in_pre_allocated_header(self, dst_path: Path):
        header_content = self._total_data_bytes.to_bytes(type(self).DATA_HEAD_SIZE_IN_BYTES, byteorder="big")
        header_content = np.frombuffer(header_content, dtype="uint8")
        # write the header content to the packed dataset file
        m = np.memmap(dst_path, mode="r+", offset=0, shape=(type(self).DATA_HEAD_SIZE_IN_BYTES,))
        m[:] = header_content[:]

    def _process_line(self, f: IO, line: str):
        sizes = [None] * len(self.codecs)

        for i, (codec, jq_filter) in enumerate(
            zip(self.codecs, self.jq_filters),
        ):
            # get object to encode and encode using codec
            jq_retrieved_text = jq_filter.input_text(line).first()
            bytestring = codec.encode(jq_retrieved_text)
            num_bytes = len(bytestring)

            if num_bytes == 0:
                raise ValueError("Detected Empty sample")

            # write bytestring to file and update size array
            f.write(bytestring)
            sizes[i] = num_bytes

        # update index and total number of bytes written
        self._index_list.append([self._current_offset] + sizes)
        self._total_data_bytes += sum(sizes)

        # exceeds size limit
        if (self._max_data_bytes is not None) and (self._total_data_bytes >= self._max_data_bytes):
            raise StopIteration
