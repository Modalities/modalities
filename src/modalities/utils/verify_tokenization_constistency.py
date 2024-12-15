import json
import os
import pickle
import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Callable

import sentencepiece as spm
import tqdm
from transformers import AutoTokenizer

from modalities.api import create_raw_data_index, pack_encoded_data
from modalities.dataloader.dataset import PackedMemMapDatasetBase


class TokenizerTypes(Enum):
    sentence_piece = "sentence_piece"
    hugging_face = "hugging_face"


def _run_tokenization(
    src_path: Path, index_path: Path, pbin_path: Path, eod_token: str, tokenizer_config: dict, jq_pattern: str = ".text"
):
    # create index
    create_raw_data_index(src_path=src_path, index_path=index_path)
    # run tokenization
    num_cpus = os.cpu_count()

    tokenization_config_dict = {
        "settings": {
            "src_path": src_path,
            "dst_path": pbin_path,
            "index_path": index_path,
            "jq_pattern": jq_pattern,
            "num_cpus": num_cpus,
            "eod_token": eod_token,
            "processing_batch_size": 10,
            "raw_samples_queue_size": 300,
            "processed_samples_queue_size": 300,
        },
        "tokenizer": {**tokenizer_config},
    }

    pack_encoded_data(config_dict=tokenization_config_dict)


def _verify_index(src_path: Path, index_path: Path):
    with open(src_path, "rb") as f:
        jsonl_binary_string = f.read()

    with open(src_path, "rb") as f:
        binary_string_list = f.readlines()

    with open(src_path, "r", encoding="utf-8") as f:
        string_list = f.readlines()

    with open(index_path, "rb") as f:
        jsonl_index = pickle.load(f)

    for i, (offset, length) in tqdm.tqdm(enumerate(jsonl_index), desc="Verifying index"):
        # check that the index works correctly on the binary data
        binary_string = binary_string_list[i]
        if binary_string_list[i].endswith(b"\n"):
            binary_string = binary_string[:-1]
        assert jsonl_binary_string[offset : offset + length] == binary_string

        # check that string when encoded with utf-8 matches the binary data
        string = string_list[i]
        if string.endswith("\n"):
            string = string[:-1]
        assert jsonl_binary_string[offset : offset + length] == string.encode("utf-8")


def _verify_pbin(
    src_path: Path,
    pbin_path: Path,
    eod_token_id: int,
    tokenizer: Callable[[str], list[int]],
    jsonl_text_key: str,
):
    dataset = PackedMemMapDatasetBase(raw_data_path=pbin_path, sample_key="text", load_index=True)

    with open(src_path, "r", encoding="utf-8") as f:
        string_list = f.readlines()
    string_list_tokenized = [tokenizer(json.loads(string)[jsonl_text_key]) for string in string_list]

    for i in tqdm.tqdm(range(len(dataset)), desc="Verifying pbin"):
        pbin_sample = dataset[i]["text"]
        recomputed_sample = string_list_tokenized[i]

        # make sure that only the last token is the eod token
        # and that the second last token is not the eod token
        assert pbin_sample[-1] == eod_token_id
        assert pbin_sample[-2] != eod_token_id

        # we need to check if tokenizer addas the eod token as
        # some tokenizers don't add the eod token at the end of the string
        # whereas modalities always adds the eod token at the end of the string
        if recomputed_sample[-1] != eod_token_id:
            if i == 0:
                warnings.warn("The tokenizer does not add the eod token at the end of the string!")
            assert len(pbin_sample) - 1 == len(recomputed_sample)
            assert all(pbin_sample[:-1] == recomputed_sample)
        else:
            assert len(pbin_sample) == len(recomputed_sample)
            assert all(pbin_sample == recomputed_sample)


def build_hf_tokenization_components(tokenizer_path_or_name: str, eod_token: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)

    def tokenizer_callable(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=True, max_length=51200000, padding=False, truncation=False)[
            "input_ids"
        ]

    tokenizer_config = {
        "component_key": "tokenizer",
        "variant_key": "pretrained_hf_tokenizer",
        "config": {
            "pretrained_model_name_or_path": tokenizer_path_or_name,
            "padding": False,
            "max_length": 51200000,
        },
    }

    eod_token_id = tokenizer.convert_tokens_to_ids(eod_token)
    return tokenizer_callable, tokenizer_config, eod_token_id


def build_sp_tokenization_components(tokenizer_path: Path, eod_token: str):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)

    def tokenizer_callable(text: str) -> list[int]:
        return tokenizer.Encode(text)

    tokenizer_config = {
        "component_key": "tokenizer",
        "variant_key": "pretrained_sp_tokenizer",
        "config": {
            "tokenizer_model_file": tokenizer_path,
        },
    }

    eod_token_id = tokenizer.PieceToId(eod_token)
    return tokenizer_callable, tokenizer_config, eod_token_id


def verify_tokenization_consistency(
    src_path: Path,
    eod_token: str,
    eod_token_id: int,
    tokenizer: Callable[[str], list[int]],
    tokenizer_config: dict,
    jsonl_text_key: str,
):
    """Verifies that the indexation and tokenization is consistent.
    This function applis the indexation and tokenization routines and then verifies
    that the index always captures entire samples and that the tokens in the JSON
    are correctly determined.
    For an example verification check out the test_end_to_end_indexation_and_tokenization_consistency test

    Args:
        src_path (Path): Path to the JSONL file
        eod_token (str): end of document token
        eod_token_id (int): The token id of the end of document token
        tokenizer (Callable[[str], list[int]]): Callable executing the tokenization
        tokenizer_config (dict): Tokenizer config (same as used in the tokenization entry point)
        jsonl_text_key (str): The key mapping to the text of interest in each JSON file
    """
    # run indeaxing and tokenization
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "index.idx"
        pbin_path = Path(tmp_dir) / "data.pbin"
        _run_tokenization(
            src_path=src_path,
            index_path=index_path,
            pbin_path=pbin_path,
            eod_token=eod_token,
            tokenizer_config=tokenizer_config,
            jq_pattern=f".{jsonl_text_key}",
        )

        # verify the index
        _verify_index(src_path=src_path, index_path=index_path)
        print("Index verified")
        # verify the tokenized data
        _verify_pbin(
            src_path=src_path,
            pbin_path=pbin_path,
            eod_token_id=eod_token_id,
            tokenizer=tokenizer,
            jsonl_text_key=jsonl_text_key,
        )
        print("Tokenization verified")
