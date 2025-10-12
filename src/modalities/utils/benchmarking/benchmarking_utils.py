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
    # all configs found in the experiment root directory
    ALL_CONFIGS = "all_configs"
    # If a job failed then the very same config can be rerun in a new subfolder by copying over the config file.
    # That's why, we only consider the most recent configs to determine the remaining configs.
    MOST_RECENT_CONFIGS = "most_recent_configs"
    # Configs for which we don't have any results yet.
    REMAINING_CONFIGS = "remaining_configs"
    # Configs that have to be run and for which we created new subfolders
    UPDATED_CONFIGS = "updated_configs"


class FileNames(Enum):
    RESULTS_FILE = "evaluation_results.jsonl"
    ERRORS_FILE_REGEX = "error_logs_*.log"  # Format: errors_logs_<hostname>_<local_rank>.log


def _count_jsonl_lines(jsonl_path: Path) -> int:
    # counts the number of lines in a jsonl file
    with jsonl_path.open() as f:
        return sum(1 for _ in f)


def _get_most_recent_configs(file_paths: list[Path]) -> list[Path]:
    """Filter the list of file paths to only include the most recent config files."""
    latest_configs = {}
    for file_path in file_paths:
        experiment_folder = file_path.parent
        hash_prefix, ts = experiment_folder.name.split("_", maxsplit=1)
        # assert file format: <hash_prefix>_YYYY-MM-DD__HH-MM-SS
        pattern = r"^[a-zA-Z0-9]+_\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}$"
        if not re.match(pattern, experiment_folder.name):
            raise ValueError(
                f"Invalid file format in file path: {file_path}, Expected format in parent directory "
                f"{experiment_folder.name}: DDDDDDDD_YYYY-MM-DD__HH-MM-SS"
            )
        experiment_folder_hash = experiment_folder.parent / hash_prefix
        if experiment_folder_hash not in latest_configs or ts > latest_configs[experiment_folder_hash][1]:
            latest_configs[experiment_folder_hash] = (file_path, ts)

    return [config[0] for config in latest_configs.values()]


def _is_experiment_done(
    config_file_path: Path, expected_steps: int, skip_exception_types: list[str] | None = None
) -> bool:
    # Check if the experiment is done based on the number of steps in the results file and potential error types.
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
                try:
                    error_dict = json.load(f)
                    error_type = error_dict["error"]["type"]
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse error log {error_log_path}: {e}")
                    error_type = "ErrorFileParsingError"
            error_types.append(error_type)
        # Check if any of the error types are in the skip list
        # E.g., if we want to skip OOM errors (i.e., not rerun such experiments),
        # then we can add "OutOfMemoryError" to the skip_exception_types list
        if len(set(skip_exception_types).intersection(set(error_types))) > 0:
            return True

    return False


def _update_experiment_folder(config_file_path: Path) -> Path:
    """Create a new folder for the experiment based on the config file path.
    The new folder will have the same hash as the original folder, but with a timestamp appended.

    Args:
        config_file_path (Path): The path to the config file.
    Returns:
        Path: The path to the new config file in the experiment folder.
    """
    experiment_folder_path = config_file_path.parent
    # copy the config file to a new folder
    hash_value = config_file_path.parent.name.split("_", maxsplit=1)[0]
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    new_folder_name = f"{hash_value}_{ts}"
    new_folder = experiment_folder_path.parent / new_folder_name
    new_folder.mkdir(parents=True, exist_ok=True)
    new_config_path = new_folder / config_file_path.name
    shutil.copy(config_file_path, new_config_path)
    return new_config_path


def get_current_sweep_status(
    exp_root: Path, expected_steps: int, skip_exception_types: Optional[list[str]] = None
) -> dict[str, list[Path]]:
    """Get the status of the sweep by assigning the config file paths to categories
    'all', 'most_recent', and 'remaining'.

    Args:
        exp_root (Path): The root directory of the experiment.
        expected_steps (int): The expected number of steps in the evaluation results.
        skip_exception_types (Optional[list[str]]): List of exception types to skip when checking if
            an experiment is done. A skipped experiment is considered as done in this case.
    Returns:
        dict[str, list[Path]]: A dictionary with keys 'all_configs', 'most_recent_configs', and 'remaining_configs',
            each containing a list of Path objects pointing to the respective config files.
    """

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
    # we can also skip certain exception types if specified (e.g., OOM errors)
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
    skip_exception_types: list[str],
) -> dict[str, list[Path]]:
    """List all remaining runs in the experiment root directory and optionally write them to a file.

    Args:
        exp_root (Path): The root directory of the experiment.
        expected_steps (int): The expected number of steps in the evaluation results.
        file_list_path (Optional[Path]): If provided, the list of remaining runs will be written to this file.
        skip_exception_types (Optional[list[str]]): List of exception types to skip when
            checking if an experiment is done. A skipped experiment is considered as done in this case.
    """
    file_list_dict = get_current_sweep_status(
        exp_root=exp_root, expected_steps=expected_steps, skip_exception_types=skip_exception_types
    )
    if set(file_list_dict[SweepSets.REMAINING_CONFIGS.value]) == set(file_list_dict[SweepSets.ALL_CONFIGS.value]):
        logger.info("No runs executed so far. Returning the list of all configs without creating new sub folders.")
        file_list_dict[SweepSets.UPDATED_CONFIGS.value] = file_list_dict[SweepSets.REMAINING_CONFIGS.value]
    else:
        logger.info("Some runs have been executed. Creating new sub folders for remaining configs.")
        # create new experiment folders for all remaining configs
        updated_configs = [
            _update_experiment_folder(yaml_path) for yaml_path in file_list_dict[SweepSets.REMAINING_CONFIGS.value]
        ]
        file_list_dict[SweepSets.UPDATED_CONFIGS.value] = updated_configs

    return file_list_dict
