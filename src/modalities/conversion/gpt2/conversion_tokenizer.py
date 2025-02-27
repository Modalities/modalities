from transformers import LlamaTokenizer

from modalities.tokenization.tokenizer_wrapper import PreTrainedSPTokenizer


def convert_tokenizer(tokenizer_model_path: str, output_dir: str) -> tuple[int, int, int]:
    """Converts a SentencePiece tokenizer to a Huggingface tokenizer.

    Args:
        tokenizer_model_path (str): Path to the SentencePiece tokenizer model file.
        output_dir (str): Path to the directory where the converted tokenizer will be saved.

    Returns:
        tuple[int, int]: The bos_token_id, eos_token_id and pad_token_id of the tokenizer.
    """
    sp_tokenizer = PreTrainedSPTokenizer(tokenizer_model_path)
    split_special_tokens = is_splitting_special_tokens(sp_tokenizer)
    # add_special_tokens = is_adding_special_tokens(sp_tokenizer)
    # FIXME: from_pretrained from file will not work in transformers v5 anymore:
    # FutureWarning: Calling LlamaTokenizer.from_pretrained() with the path to a single file or url
    #                is deprecated and won't be possible anymore in v5.
    #                Use a model identifier or the path to a directory instead.
    special_tokens, special_token_ids = build_special_token_args(sp_tokenizer)
    special_tokens, special_token_ids = dummy_special_tokens()
    hf_tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_model_path,
        split_special_tokens=split_special_tokens,
        **special_tokens,
        **special_token_ids,
    )
    hf_tokenizer.add_bos_token = False
    hf_tokenizer.add_eos_token = False
    hf_tokenizer.legacy = True
    # hf_tokenizer.add_prefix_space = False
    # _copy_special_tokens(sp_tokenizer, hf_tokenizer)
    hf_tokenizer.save_pretrained(output_dir)
    return hf_tokenizer.bos_token_id, hf_tokenizer.eos_token_id, hf_tokenizer.pad_token_id


def is_splitting_special_tokens(sp_tokenizer: PreTrainedSPTokenizer) -> bool:
    if (bos := sp_tokenizer.tokenizer.bos_id()) >= 0:
        test_token_id = bos
    elif (eos := sp_tokenizer.tokenizer.eos_id()) >= 0:
        test_token_id = eos
    elif (pad := sp_tokenizer.tokenizer.pad_id()) >= 0:
        test_token_id = pad
    elif (unk := sp_tokenizer.tokenizer.unk_id()) >= 0:
        test_token_id = unk
    else:
        return False
    return len(sp_tokenizer.tokenize(sp_tokenizer.tokenizer.id_to_piece(test_token_id))) > 1


def is_adding_special_tokens(sp_tokenizer: PreTrainedSPTokenizer) -> bool:
    token_ids = sp_tokenizer.tokenize("Hello!")
    if (bos := sp_tokenizer.tokenizer.bos_id()) >= 0:
        return token_ids[0] != bos
    if (eos := sp_tokenizer.tokenizer.eos_id()) >= 0:
        return token_ids[-1] != eos
    return False


def _copy_special_tokens(sp_tokenizer: PreTrainedSPTokenizer, hf_tokenizer: LlamaTokenizer):
    if (t := sp_tokenizer.tokenizer.bos_id()) >= 0:
        hf_tokenizer.bos_token_id = t
        hf_tokenizer.bos_token = sp_tokenizer.tokenizer.id_to_piece(t)
    else:
        hf_tokenizer.bos_token_id = None
    if (t := sp_tokenizer.tokenizer.eos_id()) >= 0:
        hf_tokenizer.eos_token_id = t
        hf_tokenizer.eos_token = sp_tokenizer.tokenizer.id_to_piece(t)
    else:
        hf_tokenizer.eos_token_id = None
    if (t := sp_tokenizer.tokenizer.pad_id()) >= 0:
        hf_tokenizer.pad_token_id = t
        hf_tokenizer.pad_token = sp_tokenizer.tokenizer.id_to_piece(t)
    else:
        hf_tokenizer.pad_token_id = None
    if (t := sp_tokenizer.tokenizer.unk_id()) >= 0:
        hf_tokenizer.unk_token_id = t
        hf_tokenizer.unk_token = sp_tokenizer.tokenizer.id_to_piece(t)
    else:
        hf_tokenizer.unk_token_id = None


def build_special_token_args(sp_tokenizer: PreTrainedSPTokenizer) -> tuple[dict[str, str], dict[str, int]]:
    special_tokens = {}
    special_token_ids = {}

    def _parse_special_token(token_name: str, sp_token_id: int):
        if sp_token_id >= 0:
            special_tokens[f"{token_name}_token"] = sp_tokenizer.tokenizer.id_to_piece(sp_token_id)
            special_token_ids[f"{token_name}_token_id"] = sp_token_id
        else:
            special_tokens[f"{token_name}_token"] = None
            special_token_ids[f"{token_name}_token_id"] = None

    _parse_special_token("bos", sp_tokenizer.tokenizer.bos_id())
    _parse_special_token("eos", sp_tokenizer.tokenizer.eos_id())
    _parse_special_token("pad", sp_tokenizer.tokenizer.pad_id())
    _parse_special_token("unk", sp_tokenizer.tokenizer.unk_id())

    return special_tokens, special_token_ids


def dummy_special_tokens() -> tuple[dict[str, str], dict[str, int]]:
    return (
        {"bos_token": None, "eos_token": None, "pad_token": None, "unk_token": None},
        {"bos_token_id": None, "eos_token_id": None, "pad_token_id": None, "unk_token_id": None},
    )
