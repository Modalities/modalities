from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer, PreTrainedSPTokenizer, TokenizerWrapper


def _assert_tokenization(tokenizer: TokenizerWrapper):
    text = "This is a test sentence."
    token_ids = tokenizer.tokenize(text)
    assert len(token_ids) > 0


def test_hf_tokenize():
    model_file = "data/tokenizer/hf_gpt2"
    tokenizer = PreTrainedHFTokenizer(
        pretrained_model_name_or_path=model_file, max_length=20, truncation=False, padding=False
    )
    _assert_tokenization(tokenizer)


def test_sp_tokenize():
    model_file = "data/tokenizer/sentencepiece/bpe_tokenizer.model"
    tokenizer = PreTrainedSPTokenizer(model_file=model_file)
    _assert_tokenization(tokenizer)
