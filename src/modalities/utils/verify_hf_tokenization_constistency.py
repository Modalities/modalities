import json
import os
import pickle
import tempfile
import warnings
from pathlib import Path

import tqdm
from transformers import AutoTokenizer

from modalities.api import create_raw_data_index, pack_encoded_data
from modalities.dataloader.dataset import PackedMemMapDatasetBase


def run_tokenization(
    src_path: Path, index_path: Path, pbin_path: Path, eod_token: str, hf_tokenizer_name: str, jq_pattern: str = ".text"
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
        "tokenizer": {
            "component_key": "tokenizer",
            "variant_key": "pretrained_hf_tokenizer",
            "config": {"pretrained_model_name_or_path": hf_tokenizer_name, "padding": False, "max_length": 51200000},
        },
    }

    pack_encoded_data(config_dict=tokenization_config_dict)


def verify_index(src_path: Path, index_path: Path):
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
        assert jsonl_binary_string[offset : offset + length] == binary_string_list[i][:-1]  # remove \n

        # check that string when encoded with utf-8 matches the binary data
        assert jsonl_binary_string[offset : offset + length] == string_list[i].encode("utf-8")[:-1]


def verify_pbin(src_path: Path, pbin_path: Path, eod_token: str, hf_tokenizer_name: str, text_key: str = "text"):
    dataset = PackedMemMapDatasetBase(raw_data_path=pbin_path, sample_key="text", load_index=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
    eod_token_id = tokenizer.convert_tokens_to_ids(eod_token)
    assert tokenizer.encode(eod_token, add_special_tokens=False) == [eod_token_id]

    with open(src_path, "r", encoding="utf-8") as f:
        string_list = f.readlines()
    string_list_tokenized = [tokenizer.encode(json.loads(string)[text_key]) for string in string_list]

    special_tokens = set([tokenizer.convert_tokens_to_ids(token) for token in tokenizer.special_tokens_map.values()])

    for i in tqdm.tqdm(range(len(dataset)), desc="Verifying pbin"):
        pbin_sample = dataset[i]["text"]
        recomputed_sample = string_list_tokenized[i]

        # make sure that only the last token is the eod token
        # and that the second last token is not the eod token or any other special token
        assert pbin_sample[-1] == eod_token_id
        assert pbin_sample[-1] in special_tokens
        assert pbin_sample[-2] != eod_token_id
        assert pbin_sample[-2] not in special_tokens

        # we need to check if tokenizer addas the eod token as
        # some tokenizers don't add the eod token at the end of the string
        # whereas modalities always adds the eod token at the end of the string
        if recomputed_sample[-1] != eod_token_id:
            assert recomputed_sample[-1] not in special_tokens
            if i == 0:
                warnings.warn(
                    f"The tokenizer {tokenizer.name_or_path} does not add the eod token at the end of the string!"
                )
            assert len(pbin_sample) - 1 == len(recomputed_sample)
            assert all(pbin_sample[:-1] == recomputed_sample)
        else:
            assert len(pbin_sample) == len(recomputed_sample)
            assert all(pbin_sample == recomputed_sample)


def verify_tokenization_consistency(src_path: Path, eod_token: str, hf_tokenizer_name: str):
    # run indeaxing and tokenization
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "index.idx"
        pbin_path = Path(tmp_dir) / "data.pbin"
        run_tokenization(src_path, index_path, pbin_path, eod_token, hf_tokenizer_name)

        # verify the index
        verify_index(src_path=src_path, index_path=index_path)
        print("Index verified")
        # verify the tokenized data
        verify_pbin(src_path=src_path, pbin_path=pbin_path, eod_token=eod_token, hf_tokenizer_name=hf_tokenizer_name)
        print("Tokenization verified")


if __name__ == "__main__":
    _src_path = Path("/raid/s3/opengptx/max_lue/modalities/tmp_data/00135.jsonl")
    _eod_token = "<|endoftext|>"  # "</s>"
    _hf_tokenizer_name = "gpt2"  # "xlm-roberta-large"
    verify_tokenization_consistency(src_path=_src_path, eod_token=_eod_token, hf_tokenizer_name=_hf_tokenizer_name)
