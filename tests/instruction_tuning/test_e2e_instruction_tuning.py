from pathlib import Path

from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import (
    PackedMemMapDatasetContinuousConfig,
    PreTrainedHFTokenizerConfig,
    ProcessGroupBackendType,
)
from modalities.config.instantiation_models import (
    InstructionTuningDataInstantiationModel,
    TrainingComponentsInstantiationModel,
)
from modalities.dataloader.apply_chat_template import split_and_apply_chat_template
from modalities.dataloader.create_instruction_tuning_data import (
    create_partitioned_instruction_tuning_index_and_pbin_files,
)
from modalities.dataloader.dataset_factory import DatasetFactory
from modalities.running_env.cuda_env import CudaEnv
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer
from tests.conftest import _ROOT_DIR


def test_e2e_instruction_tuning(monkeypatch, tmp_path):
    """
    End-to-end test for instruction tuning training. Takes ~25 seconds to run.
    This test prepares the data, applies the chat template, and runs the training.
    It verifies that a model checkpoint is created in the specified directory.
    """
    created_files = data_preperation(tmp_path)
    check_correct_packing(created_files)
    training(monkeypatch, tmp_path, created_files)


def data_preperation(tmp_path) -> list[Path]:
    """
    Run the instruction-tuning data preparation and verify that the data was prepared correctly
    """
    # this writes into the out_files directory
    config_file_path = _ROOT_DIR / Path("tests/instruction_tuning/files/apply_chat_template_config.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config_dict["settings"]["dst_path"] = tmp_path / "lorem_ipsum_instruct_converted.jsonl"

    partition_to_output_file_path_mapping = split_and_apply_chat_template(config_file_path, config_dict)

    config = InstructionTuningDataInstantiationModel(**config_dict)
    create_partitioned_instruction_tuning_index_and_pbin_files(config, partition_to_output_file_path_mapping)

    created_files = list(Path(list(Path(tmp_path).glob("*"))[0]).glob("*"))
    assert len(created_files) > 0, "No files were created during data preparation."
    for suffix in [".jsonl", ".idx", ".pbin"]:
        assert 3 == [path.suffix for path in created_files].count(
            suffix
        ), f"Expected 3 {suffix} files to be created, but found a different number."
    return created_files


def check_correct_packing(created_files: list[Path]) -> None:
    pbin_test_file_path = [file for file in created_files if file.suffix == ".pbin" and "test" in file.name][0]
    dataset = DatasetFactory.get_packed_mem_map_dataset_continuous(
        **PackedMemMapDatasetContinuousConfig(
            raw_data_path=str(pbin_test_file_path),
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
        + f"Got sample starts: {[sample[:20] for sample in samples[:5]]}"
    )


def training(monkeypatch, tmp_path, created_files: list[Path]) -> None:
    """
    Run the instruction-tuning training and verify that a model checkpoint was created.
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9949")

    # Load config
    config_path = _ROOT_DIR / Path("tests/instruction_tuning/files/config_lorem_ipsum_instruct_fsdp1.yaml")
    config_dict = load_app_config_dict(config_path, experiment_id="test_e2e_instruction_tuning")

    # Adapt config for test
    checkpointing_path = tmp_path / "instruct_checkpoints/"
    config_dict["settings"]["paths"]["checkpoint_saving_path"] = checkpointing_path.__str__()
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
        "checkpoint_path"
    ] = checkpointing_path.__str__()
    for partition in ["train", "test"]:
        # find train pbin in created files
        pbin_file_path = [file for file in created_files if file.suffix == ".pbin" and partition in file.name][0]
        config_dict = _recursive_overwrite_resolved_config(
            # this string is written as placeholder in the config file
            value_to_override=f"replace by {partition} temp path",
            new_value=str(pbin_file_path),
            config_dict=config_dict,
        )
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main = Main(config_path)
        main.config_dict = config_dict
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)

    checkpoint_files = [
        ("model" in path.name or "optimizer" in path.name) and path.suffix == ".bin"
        for path in list(checkpointing_path.glob("*"))[0].glob("*")
    ]
    assert sum(checkpoint_files) == 2, "Output of the test i.e. a model checkpoint and optimizer state was not created!"


def _recursive_overwrite_resolved_config(value_to_override: str, new_value: str, config_dict: dict) -> dict:
    """
    Recursively override the value in the config dictionary.
    """
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = _recursive_overwrite_resolved_config(value_to_override, new_value, value)
        elif isinstance(value, str) and value == value_to_override:
            config_dict[key] = new_value
    return config_dict
