import shutil
import tempfile
from contextlib import contextmanager
from typing import Iterable

from transformers import LlamaTokenizer

from modalities.tokenization.tokenizer_wrapper import PreTrainedSPTokenizer


def convert_tokenizer(tokenizer_model_path: str, output_dir: str) -> tuple[int, int, int, int]:
    """Converts a SentencePiece tokenizer to a Huggingface tokenizer.

    Args:
        tokenizer_model_path (str): Path to the SentencePiece tokenizer model file.
        output_dir (str): Path to the directory where the converted tokenizer will be saved.

    Returns:
        tuple[int, int, int, int]: The actual bos_token_id, eos_token_id, pad_token_id and
                                   unk_token_id of the tokenizer. Note, that these are not
                                   set in the transformers part of the created tokenizer.
                                   Only in the wrapped SentencePiece tokenizer.
    """
    sp_tokenizer = PreTrainedSPTokenizer(tokenizer_model_path)
    split_special_tokens = _is_splitting_special_tokens(sp_tokenizer)
    # Setting the special tokens to None in order to let the internal SentencePiece tokenizer handle them.
    special_tokens, special_token_ids = _dummy_special_tokens()
    with _create_tokenizer_directory(tokenizer_model_path) as tokenizer_model_dir:
        hf_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_model_dir, split_special_tokens=split_special_tokens, **special_tokens, **special_token_ids
        )
    hf_tokenizer.add_bos_token = False
    hf_tokenizer.add_eos_token = False
    # With legacy=True the LlamaTokenizer tokenizer will use the SentencePiece tokenizer for tokenization
    # and not run any special token logic.
    hf_tokenizer.legacy = True
    hf_tokenizer.save_pretrained(output_dir)
    return (
        sp_tokenizer.tokenizer.bos_id(),
        sp_tokenizer.tokenizer.eos_id(),
        sp_tokenizer.tokenizer.pad_id(),
        sp_tokenizer.tokenizer.unk_id(),
    )


def _is_splitting_special_tokens(sp_tokenizer: PreTrainedSPTokenizer) -> bool:
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


@contextmanager
def _create_tokenizer_directory(tokenizer_model_path: str) -> Iterable[str]:
    """Copies the tokenizer model to a temporary directory and yields the path to the model.
       The model is moved to a temporary directory because the from_pretrained method of
       the LlamaTokenizer class requires a directory path instead of a file path from transformers v5 on.
       When the returned iterator is exhausted, the temporary directory is deleted.

    Args:
        tokenizer_model_path (str): Path to the tokenizer model file.

    Yields:
        Iterable[str]: Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copy(tokenizer_model_path, f"{temp_dir}/tokenizer.model")
        yield temp_dir


def _dummy_special_tokens() -> tuple[dict[str, str], dict[str, int]]:
    return (
        {"bos_token": None, "eos_token": None, "pad_token": None, "unk_token": None},
        {"bos_token_id": None, "eos_token_id": None, "pad_token_id": None, "unk_token_id": None},
    )
