import re
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path


class FileNames(Enum):
    CONFIG_FILE = "config.yaml"
    RESULTS_FILE = "evaluation_results.jsonl"
    ERRORS_FILE = "errors.jsonl"


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


def _is_experiment_successful(config_file_path: Path, expected_steps: int) -> bool:
    """Check if the experiment is successful based on the number of steps in the results file."""
    results_path = config_file_path.parent / FileNames.RESULTS_FILE.value
    if not results_path.exists():
        return False  # Missing outputs
    steps_found = _count_jsonl_lines(results_path)
    return steps_found == expected_steps


def _keep_or_update_experiment_folder(config_file_path: Path):
    experiment_folder_path = config_file_path.parent
    error_log_path = experiment_folder_path / FileNames.ERRORS_FILE.value
    results_log_path = experiment_folder_path / FileNames.RESULTS_FILE.value
    if not error_log_path.exists() and not results_log_path.exists():
        # No errors and no results, keep the folder
        return config_file_path
    else:
        # copy the config file to a new folder
        hash = config_file_path.parent.name.split("_", maxsplit=1)[0]
        ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        new_folder_name = f"{hash}_{ts}"
        new_folder = experiment_folder_path.parent / new_folder_name
        new_folder.mkdir(parents=True, exist_ok=True)
        new_config_path = new_folder / config_file_path.name
        shutil.copy(config_file_path, new_config_path)
        return new_config_path


def list_missing_runs(exp_root: Path, file_list_path: Path, expected_steps: int):
    exp_root = exp_root.resolve()

    # Find all candidate config files and filter out resolved configs
    candidate_configs = list(exp_root.glob("**/*.yaml"))
    candidate_configs = [yaml_path for yaml_path in candidate_configs if not yaml_path.name.endswith(".resolved.yaml")]

    print("=========ALL============")
    for config in candidate_configs:
        print(config)

    # filter only most recent configs
    candidate_configs = _get_most_recent_configs(candidate_configs)

    print("=========MOST=RECENT============")
    for config in candidate_configs:
        print(config)

    # filter non-successful experiments
    candidate_configs = [
        yaml_path for yaml_path in candidate_configs if not _is_experiment_successful(yaml_path, expected_steps)
    ]

    print("=========NON=SUCCESSFUL============")
    for config in candidate_configs:
        print(config)

    # keep experiment folders that have not been run yet and create
    # new experiment folders for those that have been run but failed
    candidate_configs = [_keep_or_update_experiment_folder(yaml_path) for yaml_path in candidate_configs]

    print("=========UPDATED============")
    for config in candidate_configs:
        print(config)

    # Write the config list
    with file_list_path.open("w", encoding="utf-8") as f:
        for cfg in candidate_configs:
            f.write(str(cfg) + "\n")
    print(f"Wrote {len(candidate_configs)} config paths to {file_list_path}")
