import numpy as np
import pytest

from modalities.config.config import PreTrainedHFTokenizerConfig
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer, PreTrainedSPTokenizer, TokenizerWrapper


def _assert_tokenization(tokenizer: TokenizerWrapper):
    text = "This is a test sentence."
    token_ids = tokenizer.tokenize(text)
    assert len(token_ids) > 0


@pytest.mark.parametrize(
    "text,tokenizer_config,expected_length,expected_num_padding_tokens",
    [
        # Test cases 1: Sequence is shorter than max_length, i.e., len(text) < max_length
        # If padding="max_length", we want a sequence to be padded to the max_length,
        # irrespective of the truncation flag and only if max_length is specified.
        # If max_length is not specified, we pad to the max model input length (i.e., 1024 for the gpt2 model).
        # NOTE: "AAAAAAAA" is a single token for the gpt2 tokenizer, there is
        # no "A" sequence longer than that in the vocabulary.
        (
            "AAAAAAAA" * 6,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding="max_length",
                max_length=10,
                special_tokens={"pad_token": "[PAD]"},
            ),
            10,
            4,
        ),
        (
            "AAAAAAAA" * 6,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding="max_length",
                max_length=10,
                special_tokens={"pad_token": "[PAD]"},
            ),
            10,
            4,
        ),
        (
            "AAAAAAAA" * 6,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding="max_length",
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            1024,
            1018,
        ),
        (
            "AAAAAAAA" * 6,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding="max_length",
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            1024,
            1018,
        ),
        # If padding=False, we want no padding to be applied, irrespective of the truncation flag and max_length.,
        # irrespective of the truncation flag
        (
            "AAAAAAAA" * 6,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding=False,
                max_length=10,
                special_tokens={"pad_token": "[PAD]"},
            ),
            6,
            0,
        ),
        (
            "AAAAAAAA" * 6,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding=False,
                max_length=10,
                special_tokens={"pad_token": "[PAD]"},
            ),
            6,
            0,
        ),
        # NOTE: This is the setting used for pretraining dataset tokenisation!!!
        (
            "AAAAAAAA" * 6,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding=False,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            6,
            0,
        ),
        (
            "AAAAAAAA" * 6,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding=False,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            6,
            0,
        ),
        # Test cases 2: Sequence is longer than max_length, i.e., len(text) > max_length
        # If truncation=True and len(text)<model max length, we want a sequence
        # to be truncated to the max_length, irrespective of the padding flag.
        (
            "AAAAAAAA" * 15,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding=False,
                max_length=10,
                special_tokens={"pad_token": "[PAD]"},
            ),
            10,
            0,
        ),
        (
            "AAAAAAAA" * 15,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding=True,
                max_length=10,
                special_tokens={"pad_token": "[PAD]"},
            ),
            10,
            0,
        ),
        (
            "AAAAAAAA" * 15,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding=False,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            15,
            0,
        ),
        (
            "AAAAAAAA" * 15,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding=True,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            15,
            0,
        ),
        # If truncation=False and len(text)<model max length,
        # we want a sequence to be unmodified, irrespective of the padding flag.
        (
            "AAAAAAAA" * 15,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding=False,
                max_length=10,
                special_tokens={"pad_token": "[PAD]"},
            ),
            15,
            0,
        ),
        (
            "AAAAAAAA" * 15,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding=True,
                max_length=10,
                special_tokens={"pad_token": "[PAD]"},
            ),
            15,
            0,
        ),
        (
            "AAAAAAAA" * 15,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding=False,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            15,
            0,
        ),
        (
            "AAAAAAAA" * 15,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding=True,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            15,
            0,
        ),
        # Test cases 3: Sequence is longer than max model input length, i.e., len(text) > max model input length
        # NOTE: This is a typical case when tokenising the pretraining dataset!!!
        (
            "AAAAAAAA" * 1030,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding=False,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            1030,
            0,
        ),
        (
            "AAAAAAAA" * 1030,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=False,
                padding=True,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            1030,
            0,
        ),
        (
            "AAAAAAAA" * 1030,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding=True,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            1024,
            0,
        ),
        # if we want to pad to max model input length we have to use padding="max_length",
        # otherwise we will only pad to the longest sequence in the batch.
        # see: https://huggingface.co/docs/transformers/pad_truncation#padding-and-truncation
        (
            "AAAAAAAA" * 1020,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding=True,
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            1020,
            0,
        ),
        (
            "AAAAAAAA" * 1020,
            PreTrainedHFTokenizerConfig(
                pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
                truncation=True,
                padding="max_length",
                max_length=None,
                special_tokens={"pad_token": "[PAD]"},
            ),
            1024,
            4,
        ),
    ],
)
def test_hf_tokenize(
    text: str,
    tokenizer_config: PreTrainedHFTokenizerConfig,
    expected_length: int,
    expected_num_padding_tokens: int,
):
    # also see here for the truncation and padding options and their effects:
    # https://huggingface.co/docs/transformers/pad_truncation#padding-and-truncation

    tokenizer_config_dict = tokenizer_config.model_dump()
    tokenizer = PreTrainedHFTokenizer(**tokenizer_config_dict)

    token_ids = tokenizer.tokenize(text)

    # make sure that the overall token sequence length is correct
    assert len(token_ids) == expected_length

    # check number of non-padding tokens (token_id = 43488 corresponds to "AAAAAAAA")
    assert sum(np.array(token_ids) == 43488) == (expected_length - expected_num_padding_tokens)

    # check number of padding tokens
    assert sum(np.array(token_ids) == 50257) == expected_num_padding_tokens


@pytest.mark.skip(reason="Missing pretrained unigram sp tokenizer.")
def test_sp_tokenize():
    tokenizer_model_file = "data/tokenizer/opengptx_unigram/unigram_tokenizer.model"
    tokenizer = PreTrainedSPTokenizer(tokenizer_model_file=tokenizer_model_file)
    _assert_tokenization(tokenizer)
