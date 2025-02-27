from pathlib import Path

import pytest
from transformers import LlamaTokenizer

from modalities.conversion.gpt2.conversion_tokenizer import convert_tokenizer
from modalities.tokenization.tokenizer_wrapper import PreTrainedSPTokenizer


def test_converted_tokenizer_produces_same_tokens_as_original(
    converted_tokenizer: LlamaTokenizer, sp_tokenizer: PreTrainedSPTokenizer, text: str
):
    converted_token_ids = converted_tokenizer(text)
    sp_token_ids = sp_tokenizer.tokenize(text)
    assert converted_token_ids["input_ids"] == sp_token_ids


def test_converted_tokenizer_detokenizes_same_as_original(
    converted_tokenizer: LlamaTokenizer, sp_tokenizer: PreTrainedSPTokenizer, token_ids: list[int]
):
    converted_tokens = converted_tokenizer.decode(token_ids)
    sp_tokens = sp_tokenizer.decode(token_ids)
    assert converted_tokens == sp_tokens


@pytest.fixture
def converted_tokenizer(tmp_path: Path, tokenizer_model_file: str) -> LlamaTokenizer:
    convert_tokenizer(tokenizer_model_path=tokenizer_model_file, output_dir=tmp_path)
    return LlamaTokenizer.from_pretrained(tmp_path)


@pytest.fixture
def sp_tokenizer(tokenizer_model_file: str) -> PreTrainedSPTokenizer:
    return PreTrainedSPTokenizer(tokenizer_model_file=tokenizer_model_file)


@pytest.fixture
def tokenizer_model_file() -> str:
    return "data/tokenizer/sentencepiece_dclm/en_32k_tokenizer.model"


@pytest.fixture(
    params=[
        "<unk><s></s><pad><eod><placeholder_tok_0><placeholder_tok_1><placeholder_tok_2>",
        "<s>Hello,\n my dog is cute",
        "the secret phrase is ossifrage",
    ]
)
def text(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(
    params=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 20527, 1, 20527, 2, 20527, 3, 20527, 4, 20527, 5, 20527, 6, 20527],
        [1, 20527],
    ]
)
def token_ids(request: pytest.FixtureRequest) -> list[int]:
    return request.param
