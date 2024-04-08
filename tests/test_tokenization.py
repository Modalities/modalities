import pytest

from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer, PreTrainedSPTokenizer, TokenizerWrapper


def _assert_tokenization(tokenizer: TokenizerWrapper):
    text = "This is a test sentence."
    token_ids = tokenizer.tokenize(text)
    assert len(token_ids) > 0


def test_hf_tokenize():
    tokenizer_model_file = "data/tokenizer/hf_gpt2"
    tokenizer = PreTrainedHFTokenizer(
        pretrained_model_name_or_path=tokenizer_model_file, max_length=20, truncation=False, padding=False
    )
    _assert_tokenization(tokenizer)


@pytest.mark.skip(reason="Missing pretrained unigram sp tokenizer.")
def test_sp_tokenize():
    tokenizer_model_file = "data/tokenizer/opengptx_unigram/unigram_tokenizer.model"
    tokenizer = PreTrainedSPTokenizer(tokenizer_model_file=tokenizer_model_file)
    _assert_tokenization(tokenizer)
