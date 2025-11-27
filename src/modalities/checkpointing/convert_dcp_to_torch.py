import os
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from modalities.config.config import load_app_config_dict, save_yaml_config_dict


def convert_dcp_to_torch(dcp_checkpoint_dir: str, output_dir: str, model_key: str = "model_raw") -> str:
    """Converts a FSDP2 checkpoint to a standard PyTorch checkpoint.

    Args:
        dcp_checkpoint_dir (str): Directory containing the FSDP2 checkpoint files.
        output_dir (str): Directory to save the converted PyTorch checkpoint.
        model_key (str): Key of the model configuration in the modalities config.
    Returns:
        str: Path to the converted config file.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch_checkpoint_file = os.path.join(output_dir, "pytorch_model.bin")
    torch_config_file = convert_config_file(dcp_checkpoint_dir, output_dir, model_key, torch_checkpoint_file)
    # TODO This is the (adapted) code from torch's dcp_to_torch_save(dcp_checkpoint_dir, torch_checkpoint_file)
    #      since we only want to convert the model state dict here. In future torch versions this function might
    #      support converting only parts of the checkpoint.
    #      (from torch.distributed.checkpoint.format_utils import dcp_to_torch_save)
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd, storage_reader=FileSystemReader(dcp_checkpoint_dir), planner=_EmptyStateDictLoadPlanner(), no_dist=True
    )
    torch.save(sd["app"]["model"], torch_checkpoint_file)
    return torch_config_file


def convert_config_file(dcp_checkpoint_dir: str, output_dir: str, model_key: str, torch_checkpoint_file: str) -> str:
    """Converts the modalities config file for DCP to a config file for standard PyTorch checkpoint loading.
    Args:
        dcp_checkpoint_dir (str): Directory containing the DCP checkpoint files.
        output_dir (str): Directory to save the converted config file.
        model_key (str): Key of the model configuration in the modalities config.
        torch_checkpoint_file (str): Path to the converted PyTorch checkpoint file.
    Returns:
        str: Path to the converted config file.
    """
    config_src: str | None = find_yaml_config_in_dir(dcp_checkpoint_dir)
    if config_src is None:
        config_src = find_yaml_config_in_dir(os.path.join(dcp_checkpoint_dir, ".."))
    if config_src is None:
        raise FileNotFoundError("No YAML config file found in checkpoint directory or its parent.")
    config_dst: str = os.path.join(output_dir, os.path.basename(config_src))
    dcp_config: dict[str, Any] = load_app_config_dict(Path(config_src), experiment_id="-1")
    torch_config: dict[str, Any] = {
        "checkpointed_model": {
            "component_key": "model",
            "variant_key": "fsdp1_checkpointed",
            "config": {
                "checkpoint_loading": {
                    "component_key": "checkpoint_loading",
                    "variant_key": "torch",
                    "config": {
                        "device": 0,
                        "precision": "BF16",  # FIXME Should this be configurable?
                    },
                },
                "model": {
                    "instance_key": "model",
                    "pass_type": "BY_REFERENCE",
                },
                "checkpoint_path": torch_checkpoint_file,
            },
        },
    }
    torch_config["model"] = dcp_config[model_key]
    torch_config["model"]["config"]["use_meta_device"] = False
    save_yaml_config_dict(torch_config, config_dst)
    return config_dst


def find_yaml_config_in_dir(directory: str) -> str | None:
    """Finds the first YAML config file in the given directory.

    Args:
        directory (str): Directory to search for YAML files.

    Returns:
        str | None: Path to the found YAML file or None if not found.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            return os.path.join(directory, filename)
    return None
