from pathlib import Path

import pytest
from modalities.__main__ import entry_point_data_create_raw_index, entry_point_pack_encoded_data
from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from transformers import GPT2Tokenizer
from modalities.dataloader.apply_chat_template import apply_chat_template
from pydantic import FilePath
import json


@pytest.fixture
def tokenizer_path() -> str:
    return "/raid/s3/opengptx/alexj/llm_gym/models/SmolLM-1.7B/"


@pytest.fixture
def apply_chat_template_config_path() -> str:
    return "/raid/s3/opengptx/maxr/ogptx/modalities/config_files/data_preparation/apply_chat_template_config_for_lora_test.yaml"


@pytest.fixture
def pbin_config_path() -> str:
    return "/raid/s3/opengptx/maxr/ogptx/modalities/config_files/data_preparation/packed_chat_dataset_config_for_lora_test.yaml"


def test_if_include_to_loss_tokens_are_only_one_token(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    b_include_to_loss_token = "^"
    e_include_to_loss_token = "$"
    tokenized_output = tokenizer(b_include_to_loss_token)
    print(tokenized_output)
    assert len(tokenized_output["input_ids"]) == 1
    tokenized_output = tokenizer(e_include_to_loss_token)
    print(tokenized_output)
    assert len(tokenized_output["input_ids"]) == 1


def read_first_line_from_json(file_path: str):
    with open(file_path, "r") as file:
        first_line = file.readline()
        first_line_data = json.loads(first_line)
    return first_line_data


def test_conversation_into_prompt_conversion(apply_chat_template_config_path):
    # apply_chat_template(Path(apply_chat_template_config_path))
    original_path = "/raid/s3/opengptx/alignment_data/data/honey/data/en/ultrachat_200k_fastchat.jsonl"
    dst_path = "/raid/s3/opengptx/alignment_data/data/honey/data/en/ultrachat_200k_fastchat_converted.44a6969.jsonl"
    original_first_line = read_first_line_from_json(original_path)
    dst_first_line = read_first_line_from_json(dst_path)
    assert not original_first_line.get("chat")
    assert dst_first_line.get("chat")


def test_indexing():
    src_path = Path(
        "/raid/s3/opengptx/alignment_data/data/honey/data/en/ultrachat_200k_fastchat_converted.44a6969.jsonl"
    )
    index_path = Path("/raid/s3/opengptx/maxr/ogptx/modalities/data/lorem_ipsum_sft_converted.44a6969.idx")
    index_path = LargeFileLinesReader.default_index_path(src_path, index_path)
    if index_path.exists():
        raise ValueError("index already exists. delete it or specify different output folder.")
    print(f"reading raw data from {src_path}")
    print(f"writing index to {index_path}")
    generator = IndexGenerator(src_path)
    generator.create_index(index_path)
    print("done")


if __name__ == "__main__":
    ...
