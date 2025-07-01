from modalities.config.config import PackedMemMapDatasetContinuousConfig, PreTrainedHFTokenizerConfig
from modalities.dataloader.dataset_factory import DatasetFactory
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer


def test_qwen():
    dataset = DatasetFactory.get_packed_mem_map_dataset_continuous(
        **PackedMemMapDatasetContinuousConfig(
            raw_data_path="tests/instruction_tuning/files/smol-smoltalk_train_first_10K_converted_test.5aeeb0e.pbin",
            sequence_length=8192,
            sample_key="sample",
            reuse_last_target=False,
        ).model_dump()
    )
    tokenizer = PreTrainedHFTokenizer(
        **PreTrainedHFTokenizerConfig(
            pretrained_model_name_or_path="Qwen/Qwen2.5-0.5B",
            padding=False,
            truncation=False,
            special_tokens={
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
            },
        ).model_dump()
    )
    samples = [tokenizer.decode(entry[dataset.sample_key]) for entry in dataset]
    assert all(
        sample.startswith("You are Mody") for sample in samples
    ), "All samples should start with 'You are Mody' after applying the chat template and correct packing."


def test_gpt2():
    dataset = DatasetFactory.get_packed_mem_map_dataset_continuous(
        **PackedMemMapDatasetContinuousConfig(
            raw_data_path="tests/instruction_tuning/files/lorem_ipsum_instruct_converted_test.fd7f8dd.pbin",
            sequence_length=2048,
            sample_key="sample",
            reuse_last_target=False,
        ).model_dump()
    )
    tokenizer = PreTrainedHFTokenizer(
        **PreTrainedHFTokenizerConfig(
            pretrained_model_name_or_path="data/tokenizer/hf_gpt2",
            padding=False,
            truncation=False,
            special_tokens={"pad_token": "<|endoftext|>", "additional_special_tokens": ["^", "$", "°"]},
        ).model_dump()
    )
    samples = [tokenizer.decode(entry[dataset.sample_key]) for entry in dataset]
    assert all(sample.startswith("You are Mody") for sample in samples), (
        "All samples should start with 'You are Mody' after applying the chat template and correct packing."
        + f"Got sample starts: {[sample[:20] for sample in samples[:5]]}"  # Show first 5 samples for debugging purposes
    )
