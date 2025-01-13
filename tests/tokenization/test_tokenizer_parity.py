from pathlib import Path

import pytest
import sentencepiece as spm

from modalities.config.config import load_app_config_dict
from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapterConfig, HFTokenizerAdapter


# Tokenize using SentencePiece
def tokenize_with_sp(sp_tokenizer, text):
    tokens = sp_tokenizer.encode(text, out_type=str)
    token_ids = sp_tokenizer.encode(text, out_type=int)
    decoded_text = sp_tokenizer.decode(token_ids)
    return tokens, token_ids, decoded_text


# Tokenize using Hugging Face
def tokenize_with_hf(hf_tokenizer, text):
    tokens = hf_tokenizer.tokenize(text)
    token_ids = hf_tokenizer.encode(text, add_special_tokens=False)
    decoded_text = hf_tokenizer.decode(token_ids)
    return tokens, token_ids, decoded_text


# Tokenize using the wrapper tokenizer
def tokenize_with_wrapper(wrapper_tokenizer, text):
    tokens = wrapper_tokenizer.tokenize(text)
    token_ids = wrapper_tokenizer.encode(text)
    decoded_text = wrapper_tokenizer.decode(token_ids)
    return tokens, token_ids, decoded_text


# Load SentencePiece tokenizer
def load_sp_tokenizer(sp_model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    return sp


@pytest.fixture
def sp_tokenizer_path():
    return "tests/tokenization/tokenizer_files/sp_tokenizer/en_32k_tokenizer.model"


# Fixtures for tokenizers
@pytest.fixture
def sp_tokenizer(sp_tokenizer_path):
    tokenizer = load_sp_tokenizer(sp_tokenizer_path)
    return tokenizer


@pytest.fixture
def hf_tokenizer_path():
    return "tests/tokenization/tokenizer_files/converted_to_hf_tokenizer"


@pytest.fixture
def hf_tokenizer(hf_tokenizer_path):
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_tokenizer_path)
    return tokenizer


@pytest.fixture()
def config_file_path() -> Path:
    return Path("tests/tokenization/tokenizer_files/modalities_config/dclm_2_7B_50B_continue.yaml")


@pytest.fixture()
def config_dict(config_file_path: Path) -> dict:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture
def wrapper_tokenizer(config_dict):
    config_adapter = HFModelAdapterConfig(config=config_dict)
    tokenizer = HFTokenizerAdapter(config=config_adapter)
    return tokenizer



# Parametrized test function
@pytest.mark.parametrize("text", [
    "This is a simple sentence with punctuation! How does it handle commas, semicolons, and exclamation marks?",
    "URLs like https://www.example.com or ftp://server.org/test are quite common.",
    "Emojis are fun 😄🤔👍, but how do tokenizers handle them?",
    "Foreign languages: Bonjour! ¿Cómo estás? こんにちは! How do you handle multilingual text?",
    "Programming code: def tokenize(text): return text.split() # Python code as input.",
    "Special characters: ~!@#$%^&*()_+-={}|[]\\:\";'<>?,./` and spaces.",
    "Long sentence: In a land far, far away, there lived a programmer who loved tokenizers so much that they created thousands of tests, each weirder than the last, to ensure that every edge case imaginable was covered.",
    "Mathematical equations: E = mc^2 or f(x) = ax^2 + bx + c are common in technical text.",
    "Random string: ajsdkfhwjeio2340298hfsdjkf@@@!!!***.",
    "Numbers: 1234567890, 1,000,000, and 3.14159 are common in text as well.",
])
def test_tokenizations(sp_tokenizer, hf_tokenizer, wrapper_tokenizer, text):
    # Tokenize using all tokenizers
    sp_data = tokenize_with_sp(sp_tokenizer, text)
    hf_data = tokenize_with_hf(hf_tokenizer, text)
    wrapper_data = tokenize_with_wrapper(wrapper_tokenizer, text)

    sp_tokens, sp_token_ids, sp_decoded = sp_data
    hf_tokens, hf_token_ids, hf_decoded = hf_data
    wrapper_tokens, wrapper_token_ids, wrapper_decoded = wrapper_data

    # Token Equivalence
    assert sp_tokens == hf_tokens == wrapper_tokens, f"Token mismatch for text: {text}"

    # Token ID Equivalence
    assert sp_token_ids == hf_token_ids == wrapper_token_ids, f"Token ID mismatch for text: {text}"

    # Round-Trip Text Parity
    assert sp_decoded == hf_decoded == wrapper_decoded, f"Round-trip text mismatch for text: {text}"
