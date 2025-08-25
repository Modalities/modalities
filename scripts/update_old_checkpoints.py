import logging
import os
from pathlib import Path

import torch
import yaml

from modalities.config.config import load_app_config_dict

config_type = dict[str, "str | config_type"]


def update_model(old_model_config: str, new_model_config: str, new_checkpoint_path: str | None):
    old_checkpoint_path = update_config(old_model_config, new_model_config, new_checkpoint_path)
    test_loading_config(new_model_config)
    if new_checkpoint_path is not None:
        if not old_checkpoint_path:
            logging.error("No valid checkpoint path found in config file!")
            exit(1)
        update_model_state_dict(old_checkpoint_path, new_checkpoint_path)


def update_config(old_path: str, new_path: str, new_checkpoint_path: str | None) -> str | None:
    """
    Convert a configuration file from an old format to a new format.

    Args:
        old_path (str): Path to the old configuration file.
        new_path (str): Path to save the new configuration file.
        new_checkpoint_path (str | None): Path to the new checkpoint file, if applicable.

    Returns:
        str | None: The old checkpoint path if it was updated, otherwise None.
    """
    with open(old_path, "r") as old_file:
        config: config_type = yaml.safe_load(old_file)
    old_checkpoint_path = update_checkpoint_path(config, new_checkpoint_path)
    add_new_keys(config)
    remove_keys(config)
    rename_keys(config)
    with open(new_path, "w") as new_file:
        yaml.dump(config, new_file)
    return old_checkpoint_path


def update_checkpoint_path(config: config_type, new_checkpoint_path: str | None) -> str | None:
    if new_checkpoint_path is not None:
        if "checkpointed_model" in config:
            old_path = config["checkpointed_model"]["config"]["checkpoint_path"]
            config["checkpointed_model"]["config"]["checkpoint_path"] = new_checkpoint_path
            return old_path
        else:
            logging.error("'new_checkpoint_path' is set but no 'checkpointed_model' key found in configuration.")
            exit(1)
    return None


def rename_keys(config: config_type):
    model_config = config["model_raw" if "model_raw" in config else "model"]["config"]
    old_norm_keys = ["attention_norm", "ffn_norm", "lm_head_norm"]
    new_norm_keys = ["attention_norm_config", "ffn_norm_config", "lm_head_norm_config"]
    for old_key, new_key in zip(old_norm_keys, new_norm_keys):
        rename_config_key(model_config, old_key, new_key)
        rename_config_key(model_config[new_key], "variant_key", "norm_type")


def rename_config_key(config: config_type, old_key: str, new_key: str):
    """
    Rename a single key in the configuration dictionary.

    Args:
        config (dict): The configuration dictionary.
        old_key (str): The old key to be renamed.
        new_key (str): The new key name.
    """
    if old_key in config:
        config[new_key] = config.pop(old_key)
    else:
        logging.warning(f"Key '{old_key}' not found in configuration.")


def add_new_keys(config: config_type):
    model_config = config["model_raw" if "model_raw" in config else "model"]["config"]
    model_config["use_weight_tying"] = False
    model_config["use_meta_device"] = False


def remove_keys(config: config_type):
    if "evaluation_subscriber" in config and "experiment_id" in config["evaluation_subscriber"]["config"]:
        del config["evaluation_subscriber"]["config"]["experiment_id"]
    if "settings" in config and "experiment_id" in config["settings"]:
        del config["settings"]["experiment_id"]
    if (
        "checkpoint_saving" in config
        and "checkpoint_saving_execution" in config["checkpoint_saving"]["config"]
        and "experiment_id" in config["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"]
    ):
        del config["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"]["experiment_id"]


def update_model_state_dict(old_model_path: str, new_model_path: str):
    """
    Update the model state dictionary by loading the old model and saving it to the new path.

    Args:
        old_model_path (str): Path to the old model file.
        new_model_path (str): Path to the new model file.
    """
    state_dict = torch.load(old_model_path)
    if "lm_head.weight" in state_dict:
        state_dict["transformer.lm_head.weight"] = state_dict["lm_head.weight"]
        del state_dict["lm_head.weight"]
        torch.save(state_dict, new_model_path)
    else:
        logging.error("'lm_head.weight' not found in the model state dictionary.")
        if "transformer.lm_head.weight" in state_dict:
            logging.error("The model state dictionary already seems to be in the updated format.")


def test_loading_config(new_config_path: str):
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    load_app_config_dict(Path(new_config_path))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python update_old_checkpoints.py <old_model_config> <new_model_config> [new_checkpoint_path]")
        print("If only a config file conversion is needed, omit the third argument.")
        exit(1)

    old_model_config = sys.argv[1]
    new_model_config = sys.argv[2]
    new_checkpoint_path = sys.argv[3] if len(sys.argv) > 3 else None

    update_model(old_model_config, new_model_config, new_checkpoint_path)
