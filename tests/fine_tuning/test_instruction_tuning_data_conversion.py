import json
import os
from pathlib import Path

import pytest
from transformers import GPT2Tokenizer

from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.dataloader.apply_chat_template import split_and_apply_chat_template
from tests.conftest import _ROOT_DIR


@pytest.fixture
def tokenizer_path() -> str:
    return "/raid/s3/opengptx/alexj/llm_gym/models/SmolLM-1.7B/"


@pytest.fixture
def apply_chat_template_config_path() -> str:
    return _ROOT_DIR / Path("tests/fine_tuning/test_configs/apply_chat_template_config.yaml")


@pytest.fixture
def index_path() -> Path:
    return _ROOT_DIR / Path("data/lorem_ipsum_sft_converted.74d2208.idx")


@pytest.fixture
def pbin_config_path() -> str:
    return "/raid/s3/opengptx/maxr/ogptx/modalities/config_files/data_preparation/packed_chat_dataset_config_for_lora_test.yaml"


@pytest.fixture
def jsonl_file_path_before_applying_chat_template() -> Path:
    return "/raid/s3/opengptx/alexj/llm_gym/modalities/modalities/tests/fine_tuning/test_data/sampled_dataset.jsonl"


@pytest.fixture
def jsonl_file_path_after_applying_chat_template() -> str:
    return "/raid/s3/opengptx/alexj/llm_gym/modalities/modalities/tests/fine_tuning/test_data/sampled_dataset_c63ef47/ultrachat_200k_fastchat_sampled_converted_train.c63ef47.jsonl"


def test_if_include_to_loss_tokens_are_only_one_token(tokenizer_path: str):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    b_include_to_loss_token = "^"
    e_include_to_loss_token = "$"
    tokenized_output = tokenizer(b_include_to_loss_token)
    print(tokenized_output)
    assert len(tokenized_output["input_ids"]) == 1
    tokenized_output = tokenizer(e_include_to_loss_token)
    print(tokenized_output)
    assert len(tokenized_output["input_ids"]) == 1


def read_first_line_from_json(file_path: str) -> dict:
    with open(file_path, "r") as file:
        first_line = file.readline()
        first_line_data = json.loads(first_line)
    return first_line_data


def test_conversation_into_prompt_conversion(
        apply_chat_template_config_path: str,
        jsonl_file_path_before_applying_chat_template: str,
        jsonl_file_path_after_applying_chat_template: str
):
    # split_and_apply_chat_template(Path(apply_chat_template_config_path))
    original_first_line = read_first_line_from_json(jsonl_file_path_before_applying_chat_template)
    dst_first_line = read_first_line_from_json(jsonl_file_path_after_applying_chat_template)
    assert not original_first_line.get("chat")
    assert dst_first_line.get("chat")


def test_indexing(jsonl_file_path_after_applying_chat_template: str, index_path: Path):
    src_path = Path(jsonl_file_path_after_applying_chat_template)
    index_path = LargeFileLinesReader.default_index_path(src_path, index_path)
    if index_path.exists():
        raise ValueError("Index already exists. delete it or specify different output folder.")
    print(f"Reading raw data from {src_path}")
    print(f"W index to {index_path}")
    generator = IndexGenerator(src_path)
    generator.create_index(index_path)

    assert os.path.exists(index_path), "Index file not generated"

    # Removing the created index file
    os.remove(index_path)
    print(f"{index_path} has been deleted successfully.")
