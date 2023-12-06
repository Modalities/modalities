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
        jq_pattern=".text",
        max_tokens: int = None,
        max_length: int = None,
        size_in_bytes: int = 4,
        header_size_in_bytes: int = 8,
    ):
        self.src_path = Path(src_path)
        self.index_path = Path(index_path) if index_path is not None else None
        self.jq_filter = jq.compile(jq_pattern)
        self.tokenizer_file = tokenizer_file
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.size_in_bytes = size_in_bytes
        self.header_size_in_bytes = header_size_in_bytes

        reader = LargeFileLinesReader(src_path, index_path=index_path, lazy_init=False)
        self.num_samples = len(reader)

    def run(self, dst_path: str | Path):
        dst_path = Path(dst_path)

        if dst_path.exists():
            raise ValueError(f"file already exists at destination path '{dst_path}'.")

        reader = LargeFileLinesReader(self.src_path, index_path=self.index_path, lazy_init=False)
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
            for line in tqdm(reader):
                try:
                    if self.max_length:
                        tokens = tokenizer(
                            self.jq_filter.input_text(line).first(), max_length=self.max_length, truncation=True
                        )["input_ids"][: self.max_length]
                    else:
                        tokens = tokenizer(self.jq_filter.input_text(line).first())["input_ids"]
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


def create_packed_data(
    src_path: str | Path,
    dst_path: str | Path = None,
    index_path: str | Path = None,
    tokenizer_file: str = "./data/tokenizer/tokenizer.json",
    jq_pattern=".text",
    max_tokens: int = None,
    max_length: int = None,
    size_in_bytes: int = 4,
):
    raw_data_path = Path(src_path)

    if index_path is None:
        index_path = Path(raw_data_path.parent, f"{raw_data_path.stem}.idx.pkl")
    else:
        index_path = Path(index_path)

    if dst_path is None:
        dst_path = Path(raw_data_path.parent, f"{raw_data_path.stem}.packed.bin")
    else:
        dst_path = Path(dst_path)

    generator = PackedDataGenerator(
        src_path,
        index_path=index_path,
        tokenizer_file=tokenizer_file,
        jq_pattern=jq_pattern,
        max_tokens=max_tokens,
        max_length=max_length,
        size_in_bytes=size_in_bytes,
    )
    generator.run(dst_path)
