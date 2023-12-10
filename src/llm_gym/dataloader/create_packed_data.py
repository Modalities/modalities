import pickle
from pathlib import Path

import jq
import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from llm_gym.dataloader.large_file_lines_reader import LargeFileLinesReader


class PackedDataGenerator:
    def __init__(
        self,
        src_path: str | Path,
        index_path: str | Path = None,
        tokenizer_file: str = "./data/tokenizer/tokenizer.json",
        jq_pattern: str = ".text",
        max_tokens: int = None,
        max_length: int = None,
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
        :param max_length: TODO - necessary?
        :param size_in_bytes: amount of bytes to represent tokens as integers.
                              If the vocabulary exceeds 2^`size_in_bytes`, this requires adaptation.
        :param header_size_in_bytes: amount of bytes to represent number of all tokens in dataset.
                                     If the amount exceeds 2^`header_size_in_bytes`, this requires adaptation.
        """
        self.src_path = Path(src_path)
        self.jq_filter = jq.compile(jq_pattern)
        self.tokenizer_file = tokenizer_file
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.size_in_bytes = size_in_bytes
        self.header_size_in_bytes = header_size_in_bytes

        self._reader = LargeFileLinesReader(src_path, index_path=index_path)
        self.num_samples = len(self._reader)

    def run(self, dst_path: str | Path):
        dst_path = Path(dst_path)

        if dst_path.exists():
            raise ValueError(f"file already exists at destination path '{dst_path}'.")

        tokenizer = GPT2TokenizerFast(tokenizer_file=self.tokenizer_file)

        num_tokens = 0
        curr_offset = self.header_size_in_bytes
        index_list = []
        eos_token_as_bytes = tokenizer(tokenizer.eos_token)["input_ids"][0].to_bytes(
            self.size_in_bytes, byteorder="big"
        )
        with dst_path.open("wb") as f:
            # allocate first self.header_size_in_bytes bytes for header (encodes length of data section)
            f.write((0).to_bytes(self.header_size_in_bytes, byteorder="big"))

            # write data section (tokens)
            for line in tqdm(self._reader):
                try:
                    self._process_line(curr_offset, eos_token_as_bytes, f, index_list, line, num_tokens, tokenizer)
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

    def _process_line(self, curr_offset, eos_token_as_bytes, f, index_list, line, num_tokens, tokenizer):
        jq_retrieved_text = self.jq_filter.input_text(line).first()
        if self.max_length:
            tokens = tokenizer(jq_retrieved_text, max_length=self.max_length, truncation=True)["input_ids"]
        else:
            tokens = tokenizer(jq_retrieved_text)["input_ids"]
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


def create_packed_data(
    src_path: str | Path,
    dst_path: str | Path = None,
    index_path: str | Path = None,
    tokenizer_file: str = "./data/tokenizer/tokenizer.json",
    jq_pattern=".text",
    **kwargs,
):
    raw_data_path = Path(src_path)

    if index_path is None:
        index_path = Path(raw_data_path.parent, f"{raw_data_path.stem}.idx")
    else:
        index_path = Path(index_path)

    if dst_path is None:
        dst_path = Path(raw_data_path.parent, f"{raw_data_path.stem}.pbin")
    else:
        dst_path = Path(dst_path)

    generator = PackedDataGenerator(
        src_path, index_path=index_path, tokenizer_file=tokenizer_file, jq_pattern=jq_pattern, **kwargs
    )
    generator.run(dst_path)
