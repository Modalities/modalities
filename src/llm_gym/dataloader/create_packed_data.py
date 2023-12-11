import pickle
from pathlib import Path

import jq
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from llm_gym.dataloader.large_file_lines_reader import LargeFileLinesReader


class PackedDataGenerator:
    def __init__(
        self,
        src_path: Path,
        tokenizer: PreTrainedTokenizer,
        index_path: Path = None,
        jq_pattern: str = ".text",
        max_tokens: int = None,
        size_in_bytes: int = 4,
        header_size_in_bytes: int = 8,
    ):
        """
        :param raw_data_path: Path to a jsonl file, which holds text data
        :param index_path: Path to an index file, which is supposed to indicate the start character position
                           and length of samples given in `raw_data_path`.
                           If not defined, an index next to `raw_data_path` is picked,
                           by replacing its suffix with ".idx".
        :param tokenizer_file: TODO
        :param jq_pattern: jq-pattern applied on every jsonl-entry. Results are afterwards tokenized and packed
        :param max_tokens: TODO - necessary?
        :param size_in_bytes: amount of bytes to represent tokens as integers.
                              If the vocabulary exceeds 2^`size_in_bytes`, this requires adaptation.
        :param header_size_in_bytes: amount of bytes to represent number of all tokens in dataset.
                                     If the amount exceeds 2^`header_size_in_bytes`, this requires adaptation.
        """
        self.src_path = src_path
        self.tokenizer = tokenizer
        self.jq_filter = jq.compile(jq_pattern)
        self.max_tokens = max_tokens
        self.size_in_bytes = size_in_bytes
        self.header_size_in_bytes = header_size_in_bytes

        self._reader = LargeFileLinesReader(src_path, index_path=index_path)
        self.num_samples = len(self._reader)

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
        dst_path = self._default_destination_path(destination_path=dst_path)

        if dst_path.exists():
            raise ValueError(f"file already exists at destination path '{dst_path}'.")

        num_tokens = 0
        curr_offset = self.header_size_in_bytes
        index_list = []
        eos_token_as_bytes = self.tokenizer(self.tokenizer.eos_token)["input_ids"][0].to_bytes(
            self.size_in_bytes, byteorder="big"
        )
        with dst_path.open("wb") as f:
            # allocate first self.header_size_in_bytes bytes for header (encodes length of data section)
            f.write((0).to_bytes(self.header_size_in_bytes, byteorder="big"))

            # write data section (tokens)
            for line in tqdm(self._reader):
                try:
                    self._process_line(curr_offset, eos_token_as_bytes, f, index_list, line, num_tokens)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"could not process line: {e=}")

            # write index
            f.write(pickle.dumps(index_list))

        # update header
        header_data = (index_list[-1][0] + index_list[-1][1] - self.header_size_in_bytes).to_bytes(
            self.header_size_in_bytes, byteorder="big"
        )
        header_data = np.frombuffer(header_data, dtype="uint8")
        m = np.memmap(dst_path, mode="r+", offset=0, shape=(self.header_size_in_bytes,))
        m[:] = header_data[:]

    def _process_line(self, curr_offset, eos_token_as_bytes, f, index_list, line, num_tokens):
        jq_retrieved_text = self.jq_filter.input_text(line).first()
        tokens = self.tokenizer(jq_retrieved_text)["input_ids"]
        if len(tokens) != 0:
            for token_idx, token in enumerate(tokens):
                token_as_bytes = token.to_bytes(self.size_in_bytes, byteorder="big")
                f.write(token_as_bytes)
                num_tokens += 1
                if num_tokens == self.max_tokens:
                    segment_length = (token_idx + 1) * self.size_in_bytes
                    index_list.append((curr_offset, segment_length))
                    curr_offset += segment_length
                    raise StopIteration
            f.write(eos_token_as_bytes)
            segment_length = (token_idx + 2) * self.size_in_bytes
            index_list.append((curr_offset, segment_length))
            curr_offset += segment_length
