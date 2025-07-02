import shutil
from pathlib import Path

import yaml

from modalities.api import FileExistencePolicy, create_raw_data_index, pack_encoded_data
from modalities.config.config import load_app_config_dict
from modalities.config.instantiation_models import InstructionTuningDataInstantiationModel
from modalities.dataloader.apply_chat_template import split_and_apply_chat_template


def create_instruction_tuning_data(config_file_path: Path):
    """
    Create instruction tuning data by applying the chat template to the raw data.
    """
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    # split and apply chat template
    partition_to_output_file_path_mapping = split_and_apply_chat_template(config_file_path, config_dict)

    config = InstructionTuningDataInstantiationModel(**config_dict)
    create_partitioned_instruction_tuning_index_and_pbin_files(config, partition_to_output_file_path_mapping)


def create_partitioned_instruction_tuning_index_and_pbin_files(
    config: InstructionTuningDataInstantiationModel, partition_to_output_file_path_mapping: dict[str, Path]
):
    hash_suffix = list(partition_to_output_file_path_mapping.values())[0].suffixes[0]
    for partition, jsonl_data_out_file_path in partition_to_output_file_path_mapping.items():
        # create the index
        idx_file_path = jsonl_data_out_file_path.with_suffix(".idx")
        create_raw_data_index(
            jsonl_data_out_file_path, idx_file_path, file_existence_policy=FileExistencePolicy.OVERRIDE
        )

        # create pbin files
        pbin_config_file_path = jsonl_data_out_file_path.with_name(f"pbin_config_{partition}").with_suffix(
            f"{hash_suffix}.yaml"
        )
        shutil.copyfile(config.settings.pbin_creation_config_file_path, pbin_config_file_path)
        pbin_config = load_app_config_dict(config_file_path=pbin_config_file_path)
        pbin_config["settings"]["src_path"] = str(jsonl_data_out_file_path)
        pbin_config["settings"]["index_path"] = str(idx_file_path)
        pbin_config["settings"]["dst_path"] = str(idx_file_path.with_suffix(".pbin"))
        with open(pbin_config_file_path, "w", encoding="utf-8") as f:
            yaml.dump(pbin_config, f, allow_unicode=True)
        pack_encoded_data(pbin_config, file_existence_policy=FileExistencePolicy.OVERRIDE)
