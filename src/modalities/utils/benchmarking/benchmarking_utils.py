import json
import re
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from modalities.utils.logger_utils import get_logger

logger = get_logger(name="main")


class SweepSets(Enum):
    ALL_CONFIGS = "all_configs"
    MOST_RECENT_CONFIGS = "most_recent_configs"
    REMAINING_CONFIGS = "remaining_configs"
    UPDATED_CONFIGS = "updated_configs"


class FileNames(Enum):
    CONFIG_FILE = "config.yaml"
    RESULTS_FILE = "evaluation_results.jsonl"
    ERRORS_FILE_REGEX = "error_logs_*.log"  # Format: errors_logs_<hostname>_<local_rank>.log


def _count_jsonl_lines(jsonl_path: Path) -> int:
    with jsonl_path.open() as f:
        return sum(1 for _ in f)


def _get_most_recent_configs(file_paths: list[Path]) -> list[Path]:
    """Filter the list of file paths to only include the most recent config files."""
    latest_configs = {}
    for file_path in file_paths:
        experiment_folder = file_path.parent
        hash, ts = experiment_folder.name.split("_", maxsplit=1)
        # assert file format: DDDDDDDD_YYYY-MM-DD__HH-MM-SS
        pattern = r"^[a-zA-Z0-9]+_\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$"
        if not re.match(pattern, experiment_folder.name):
            raise ValueError(f"Invalid file format in file path: {file_path}")
        experiment_folder_hash = experiment_folder.parent / hash
        if experiment_folder_hash not in latest_configs or ts > latest_configs[experiment_folder_hash][1]:
            latest_configs[experiment_folder_hash] = (file_path, ts)

    return [config[0] for config in latest_configs.values()]


def _is_experiment_done(config_file_path: Path, expected_steps: int, skip_exception_types: list[str] = None) -> bool:
    """Check if the experiment is done based on the number of steps in the results file and potential error types."""
    results_path = config_file_path.parent / FileNames.RESULTS_FILE.value
    # Check if results file exists and has the expected number of steps
    if results_path.exists():
        steps_found = _count_jsonl_lines(results_path)
        if steps_found == expected_steps:
            return True
    # Check if there are any errors due to which we want to skip the experiment (e.g., OOM errors)
    if skip_exception_types is not None:
        error_log_paths = list(config_file_path.parent.glob(FileNames.ERRORS_FILE_REGEX.value))
        error_types = []
        for error_log_path in error_log_paths:
            with error_log_path.open("r", encoding="utf-8") as f:
                error_type = json.load(f)["error"]["type"]
            error_types.append(error_type)
        # Check if any of the error types are in the skip list
        if len(set(skip_exception_types).intersection(set(error_types))) > 0:
            return True

    return False


def update_experiment_folder(config_file_path: Path):
    experiment_folder_path = config_file_path.parent
    # copy the config file to a new folder
    hash = config_file_path.parent.name.split("_", maxsplit=1)[0]
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    new_folder_name = f"{hash}_{ts}"
    new_folder = experiment_folder_path.parent / new_folder_name
    new_folder.mkdir(parents=True, exist_ok=True)
    new_config_path = new_folder / config_file_path.name
    shutil.copy(config_file_path, new_config_path)
    return new_config_path


def get_current_sweep_status(
    exp_root: Path, expected_steps: int, skip_exception_types: list[str] = None
) -> dict[str, list[Path]]:
    """Get the status of the sweep by listing all configs and checking their results."""
    exp_root = exp_root.resolve()
    file_list_dict = {}
    # Find all candidate config files and filter out resolved configs
    candidate_configs = list(exp_root.glob("**/*.yaml"))
    candidate_configs = [yaml_path for yaml_path in candidate_configs if not yaml_path.name.endswith(".resolved.yaml")]
    file_list_dict[SweepSets.ALL_CONFIGS.value] = candidate_configs

    # filter only most recent configs
    candidate_configs = _get_most_recent_configs(candidate_configs)
    file_list_dict[SweepSets.MOST_RECENT_CONFIGS.value] = candidate_configs

    # filter non-successful experiments, i.e., those that do not have the
    # expected number of steps in evaluation_results.jsonl
    # we can also skip certain exception types if specified
    candidate_configs = [
        yaml_path
        for yaml_path in candidate_configs
        if not _is_experiment_done(yaml_path, expected_steps, skip_exception_types)
    ]
    file_list_dict[SweepSets.REMAINING_CONFIGS.value] = candidate_configs
    return file_list_dict


def get_updated_sweep_status(
    exp_root: Path,
    expected_steps: int,
    file_list_path: Optional[Path] = None,
    skip_exception_types: Optional[list[str]] = None,
    new_folders_for_remaining: bool = False,
) -> dict[str, list[Path]]:
    """List all remaining runs in the experiment root directory and write them to a file."""
    file_list_dict = get_current_sweep_status(
        exp_root=exp_root, expected_steps=expected_steps, skip_exception_types=skip_exception_types
    )
    if not new_folders_for_remaining or set(file_list_dict[SweepSets.REMAINING_CONFIGS.value]) == set(
        file_list_dict[SweepSets.ALL_CONFIGS.value]
    ):
        logger.info("No runs executed so far. Returning the list of all configs without creating new folders.")
        file_list_dict[SweepSets.UPDATED_CONFIGS.value] = file_list_dict[SweepSets.REMAINING_CONFIGS.value]
    else:
        logger.info("Some runs have been executed. Creating new folders for remaining configs.")
        # create new experiment folders for all remaining configs
        updated_configs = [
            update_experiment_folder(yaml_path) for yaml_path in file_list_dict[SweepSets.REMAINING_CONFIGS.value]
        ]
        file_list_dict[SweepSets.UPDATED_CONFIGS.value] = updated_configs

    # Write the config list
    if file_list_path is not None:
        with file_list_path.open("w", encoding="utf-8") as f:
            for cfg in file_list_dict[SweepSets.UPDATED_CONFIGS.value]:
                f.write(str(cfg) + "\n")

        return file_list_dict
