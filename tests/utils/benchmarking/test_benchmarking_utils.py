import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from modalities.utils.benchmarking.benchmarking_utils import SweepSets, _count_jsonl_lines, get_updated_sweep_status

EXPPECTED_STEPS = 10


class FileStructure:
    structure = {
        "4": {
            "abcd0000_2025-07-27__18-00-00": {  # unsuccessful run (unexpected error)
                "error_logs_xyz.log": "unexpected",
                "config.yaml": True,
            },
            "abcd0001_2025-07-27__18-00-00": {  # unsuccessful run (unexpected error)
                "config.yaml": True,
                "error_logs_xyz.log": "unexpected",
            },
            "abcd0001_2025-07-27__18-00-10": {  # successful run
                "config.yaml": True,
                "evaluation_results.jsonl": EXPPECTED_STEPS,
            },
        },
        "8": {
            "abcd0000_2025-07-27__18-00-00": {  # successful run
                "config.yaml": True,
                "evaluation_results.jsonl": EXPPECTED_STEPS,
            },
            "abcd0001_2025-07-27__18-00-00": {  # unsuccessful run (not enough steps)
                "config.yaml": True,
                "evaluation_results.jsonl": EXPPECTED_STEPS // 2,
            },
        },
        "16": {
            "abcd0000_2025-07-27__18-00-00": {  # successful run (expected oom)
                "error_logs_xyz.log": "oom",
                "config.yaml": True,
            },
            "abcd0001_2025-07-27__18-00-00": {"config.yaml": True},  # unsuccessful run / not run yet
        },
    }

    error_log_content = {
        "unexpected": {
            "environment": {
                "rank": 7,
                "local_rank": 3,
                "world_size": 128,
                "hostname": "some.hostname.local",
            },
            "error": {
                "error": "Unexpected Error....",
                "type": "UnexpectedError",
                "stacktrace": ["Traceback (most recent call last):\\n", "..."],
            },
        },
        "oom": {
            "environment": {
                "rank": 7,
                "local_rank": 3,
                "world_size": 128,
                "hostname": "some.hostname.local",
            },
            "error": {
                "error": "CUDA out of memory...",
                "type": "OutOfMemoryError",
                "stacktrace": ["Traceback (most recent call last):\\n", "..."],
            },
        },
    }

    config_content = {
        "config_id": "",
    }


def create_test_sweep_structure(base_dir: Path):
    def make_eval_line(step):
        return {
            "dataloader_tag": "train",
            "num_train_steps_done": step,
            "losses": {"train loss avg": 11.640625, "train loss last": 11.640625},
            "metrics": {"consumed tokens": 16384, "grad norm avg": 42.9042, "grad norm last": 42.9042},
            "throughput_metrics": {
                "train samples/s": 2.5534,
                "train mfu (16-bit)": 0.1664,
                "lr mean": 6.00299e-07,
                "peak memory rank 0 (MB)": 45201.867,
            },
        }

    for num_ranks, runs in FileStructure.structure.items():
        for run_name, files in runs.items():
            run_dir: Path = base_dir / num_ranks / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            if files.get("config.yaml"):
                config_content = copy.deepcopy(FileStructure.config_content)
                config_content["config_id"] = f"{num_ranks}_{run_name}"
                (run_dir / "config.yaml").write_text(json.dumps(config_content, indent=2))

            if isinstance(files.get("evaluation_results.jsonl"), int):
                num_rows = files["evaluation_results.jsonl"]
                rows = [json.dumps(make_eval_line(i + 1)) + "\n" for i in range(num_rows)]
                with open(run_dir / "evaluation_results.jsonl", "w", encoding="utf-8") as f:
                    f.writelines(rows)

            if "error_logs_xyz.log" in files:
                error_type = files["error_logs_xyz.log"]
                with open(run_dir / "error_logs_xyz.log", "w", encoding="utf-8") as f:
                    json.dump(FileStructure.error_log_content[error_type], f, indent=2)


def test_count_jsonl_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test.jsonl"
        lines = ["{}\n" for _ in range(5)]
        jsonl_path.write_text("".join(lines))
        assert _count_jsonl_lines(jsonl_path) == 5


def test_get_updated_sweep_status(tmp_path):
    experiments_path = tmp_path / "experiments"
    file_list_path = tmp_path / "remaining_runs.txt"
    create_test_sweep_structure(experiments_path)
    with patch("modalities.utils.benchmarking.benchmarking_utils.datetime") as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "x"
        file_list_dict = get_updated_sweep_status(
            exp_root=experiments_path,
            file_list_path=file_list_path,
            expected_steps=EXPPECTED_STEPS,
            skip_exception_types=["OutOfMemoryError"],
        )

    all_configs: list[Path] = []
    for num_ranks, runs in FileStructure.structure.items():
        for run_name in runs.keys():
            config_file_path = experiments_path / num_ranks / run_name / "config.yaml"
            all_configs.append(config_file_path)
    assert set(all_configs) == set(file_list_dict[SweepSets.ALL_CONFIGS.value])

    # remove old runs (i.e., those that have failed previously and
    # and a new experiment folder has been added already)
    most_recent_configs = set(all_configs)
    most_recent_configs.remove(experiments_path / "4/abcd0001_2025-07-27__18-00-00/config.yaml")
    assert most_recent_configs == set(file_list_dict[SweepSets.MOST_RECENT_CONFIGS.value])

    # remove successful runs leading to all remaining runs
    remaining_configs = set(most_recent_configs)
    remaining_configs.remove(experiments_path / "4/abcd0001_2025-07-27__18-00-10/config.yaml")
    remaining_configs.remove(experiments_path / "8/abcd0000_2025-07-27__18-00-00/config.yaml")
    remaining_configs.remove(experiments_path / "16/abcd0000_2025-07-27__18-00-00/config.yaml")
    assert remaining_configs == set(file_list_dict[SweepSets.REMAINING_CONFIGS.value])

    # create new experiment folders for configs that have partial results already
    # e.g., an error log or some steps in evaluation_results.jsonl
    updated_configs = set(remaining_configs)
    updated_configs.remove(experiments_path / "4/abcd0000_2025-07-27__18-00-00/config.yaml")
    updated_configs.remove(experiments_path / "8/abcd0001_2025-07-27__18-00-00/config.yaml")
    updated_configs.add(experiments_path / "4/abcd0000_x/config.yaml")
    updated_configs.add(experiments_path / "8/abcd0001_x/config.yaml")
    assert updated_configs == set(file_list_dict[SweepSets.UPDATED_CONFIGS.value])

    # check that the new config is equal to the historically failed one
    with (experiments_path / "4/abcd0000_x/config.yaml").open("r", encoding="utf-8") as f:
        config_content = json.load(f)
    assert config_content["config_id"] == "4_abcd0000_2025-07-27__18-00-00"

    with (experiments_path / "8/abcd0001_x/config.yaml").open("r", encoding="utf-8") as f:
        config_content = json.load(f)
    assert config_content["config_id"] == "8_abcd0001_2025-07-27__18-00-00"
