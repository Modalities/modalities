from pathlib import Path

import pytest
from transformers import LlamaTokenizer

from modalities.conversion.gpt2.conversion_tokenizer import convert_tokenizer
from modalities.tokenization.tokenizer_wrapper import PreTrainedSPTokenizer


def test_converted_tokenizer_produces_same_tokens_as_original(
    converted_tokenizer: LlamaTokenizer, sp_tokenizer: PreTrainedSPTokenizer, text: str
):
    converted_tokens = converted_tokenizer(text)
    sp_tokens = sp_tokenizer.tokenize(text)
    assert converted_tokens["input_ids"] == sp_tokens


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


@pytest.fixture(params=["This is a test sentence.", "Hello, my dog is cute", "the secret phrase is ossifrage"])
def text(request: pytest.FixtureRequest) -> str:
    return request.param
