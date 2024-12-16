from collections import namedtuple
from pathlib import Path

import pytest

from modalities.utils.verify_tokenization_consistency import (
    TokenizerTypes,
    build_hf_tokenization_components,
    build_sp_tokenization_components,
    verify_tokenization_consistency,
)

TokenizerSettings = namedtuple("TokenizerSettings", "tokenizer_type tokenizer_name_or_path")
gpt2_settings = TokenizerSettings(
    tokenizer_type=TokenizerTypes.hugging_face,
    tokenizer_name_or_path="gpt2",
)
xlm_roberta_large_settings = TokenizerSettings(
    tokenizer_type=TokenizerTypes.hugging_face, tokenizer_name_or_path="xlm-roberta-large"
)
sentence_piece_settings = TokenizerSettings(
    tokenizer_type=TokenizerTypes.sentence_piece,
    tokenizer_name_or_path="../data/tokenizer/sentencepiece_dclm/en_32k_tokenizer.model",
)


@pytest.mark.parametrize(
    "tokenizer_settings, src_path, jsonl_text_key, eod_token, expect_error, expected_warning",
    [
        # without errors
        # test with the actual eod token
        (gpt2_settings, Path("data/datasets/lorem_ipsum_long.jsonl"), "text", "<|endoftext|>", False, None),
        (xlm_roberta_large_settings, Path("data/datasets/lorem_ipsum_long.jsonl"), "text", "</s>", False, None),
        (sentence_piece_settings, Path("data/datasets/lorem_ipsum_long.jsonl"), "text", "</s>", False, None),
        # without \n in the last line
        (
            gpt2_settings,
            Path("data/datasets/lorem_ipsum_without_last_newline.jsonl"),
            "text",
            "<|endoftext|>",
            False,
            None,
        ),
        (
            xlm_roberta_large_settings,
            Path("data/datasets/lorem_ipsum_without_last_newline.jsonl"),
            "text",
            "</s>",
            False,
            None,
        ),
        (
            sentence_piece_settings,
            Path("data/datasets/lorem_ipsum_without_last_newline.jsonl"),
            "text",
            "</s>",
            False,
            None,
        ),
        (gpt2_settings, Path("data/datasets/danish_test_dataset.jsonl"), "text", "<|endoftext|>", False, None),
        (xlm_roberta_large_settings, Path("data/datasets/danish_test_dataset.jsonl"), "text", "</s>", False, None),
        (sentence_piece_settings, Path("data/datasets/danish_test_dataset.jsonl"), "text", "</s>", False, None),
        # we also accept tokens as eod token that are not the original eod token or any other special token
        # A normal token such as "a" will pass through. It is the users obligation to pick the correct eod token
        # for a given tokenizer. The reason is that there is no way to get this information for all tokenizer
        # implementations regarding the true eod token!
        (gpt2_settings, Path("data/datasets/lorem_ipsum_long.jsonl"), "text", "a", False, None),
        (xlm_roberta_large_settings, Path("data/datasets/lorem_ipsum_long.jsonl"), "text", "a", False, None),
        (sentence_piece_settings, Path("data/datasets/lorem_ipsum_long.jsonl"), "text", "a", False, None),
        # with errors / warnings
        # eod token is not a single token
        (
            gpt2_settings,
            Path("data/datasets/lorem_ipsum_long.jsonl"),
            "text",
            "abc123",
            False,
            "The provided eod token .* has the same token id (.*) as the unk token",
        ),
        (
            xlm_roberta_large_settings,
            Path("data/datasets/lorem_ipsum_long.jsonl"),
            "text",
            "abc123",
            False,
            "The provided eod token .* has the same token id (.*) as the unk token",
        ),
        # with errors
        # eod token is not a single token
        (sentence_piece_settings, Path("data/datasets/lorem_ipsum_long.jsonl"), "text", "abc123", True, None),
    ],
)
def test_end_to_end_indexation_and_tokenization_consistency(
    tokenizer_settings: TokenizerSettings,
    src_path: Path,
    jsonl_text_key: str,
    eod_token: str,
    expect_error: bool,
    expected_warning: str,
):
    # hf
    if tokenizer_settings.tokenizer_type == TokenizerTypes.hugging_face:
        tokenizer_callable, tokenizer_config, eod_token_id = build_hf_tokenization_components(
            tokenizer_path_or_name=tokenizer_settings.tokenizer_name_or_path,
            eod_token=eod_token,
        )
        print(f"{eod_token_id=}")

    # sentence piece
    elif tokenizer_settings.tokenizer_type == TokenizerTypes.sentence_piece:
        tokenizer_callable, tokenizer_config, eod_token_id = build_sp_tokenization_components(
            tokenizer_path=tokenizer_settings.tokenizer_name_or_path,
            eod_token=eod_token,
        )

    else:
        raise ValueError(f"Tokenizer type {tokenizer_settings.tokenizer_type} not supported!")

    if expect_error:
        with pytest.raises(Exception):
            verify_tokenization_consistency(
                src_path=src_path,
                eod_token=eod_token,
                eod_token_id=eod_token_id,
                tokenizer=tokenizer_callable,
                tokenizer_config=tokenizer_config,
                jsonl_text_key=jsonl_text_key,
            )
    elif expected_warning is not None:
        with pytest.warns(UserWarning, match=expected_warning):
            verify_tokenization_consistency(
                src_path=src_path,
                eod_token=eod_token,
                eod_token_id=eod_token_id,
                tokenizer=tokenizer_callable,
                tokenizer_config=tokenizer_config,
                jsonl_text_key=jsonl_text_key,
            )
    else:
        verify_tokenization_consistency(
            src_path=src_path,
            eod_token=eod_token,
            eod_token_id=eod_token_id,
            tokenizer=tokenizer_callable,
            tokenizer_config=tokenizer_config,
            jsonl_text_key=jsonl_text_key,
        )
