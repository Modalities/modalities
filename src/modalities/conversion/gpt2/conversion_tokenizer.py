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
    hf_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_model_path)
    hf_tokenizer.add_bos_token = False  # FIXME is this always correct?
    hf_tokenizer.add_eos_token = False
    _copy_special_tokens(sp_tokenizer, hf_tokenizer)
    hf_tokenizer.save_pretrained(output_dir)
    return hf_tokenizer.bos_token_id, hf_tokenizer.eos_token_id, hf_tokenizer.pad_token_id


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
