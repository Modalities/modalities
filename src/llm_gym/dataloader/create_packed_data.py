from pathlib import Path

import jq
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
    ):
        self.src_path = Path(src_path)
        self.index_path = Path(index_path) if index_path is not None else None
        self.jq_filter = jq.compile(jq_pattern)
        self.tokenizer_file = tokenizer_file
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.size_in_bytes = size_in_bytes

        reader = LargeFileLinesReader(src_path, index_path=index_path, lazy_init=False)
        self.num_samples = len(reader)

    def run(self, dst_path: str | Path):
        dst_path = Path(dst_path)

        if dst_path.exists():
            raise ValueError(f"file already exists at destination path '{dst_path}'.")

        reader = LargeFileLinesReader(self.src_path, index_path=self.index_path, lazy_init=False)
        tokenizer = GPT2TokenizerFast(tokenizer_file=self.tokenizer_file)

        num_tokens = 0
        eos_token_as_bytes = tokenizer(tokenizer.eos_token)["input_ids"][0].to_bytes(
            self.size_in_bytes, byteorder="big"
        )
        with dst_path.open("wb") as f:
            for line in tqdm(reader):
                try:
                    if self.max_length:
                        tokens = tokenizer(
                            self.jq_filter.input_text(line).first(), max_length=self.max_length, truncation=True
                        )["input_ids"]
                    else:
                        tokens = tokenizer(self.jq_filter.input_text(line).first())["input_ids"]
                    for token in tokens:
                        token_as_bytes = token.to_bytes(self.size_in_bytes, byteorder="big")
                        f.write(token_as_bytes)
                        num_tokens += 1
                        if num_tokens == self.max_tokens:
                            raise StopIteration
                    f.write(eos_token_as_bytes)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"could not process line: {e=}")


def create_packed_data(
    src_path: str | Path,
    dst_path: str | Path,
    index_path: str | Path = None,
    tokenizer_file: str = "./data/tokenizer/tokenizer.json",
    jq_pattern=".text",
    max_tokens: int = None,
    max_length: int = None,
    size_in_bytes: int = 4,
):
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
