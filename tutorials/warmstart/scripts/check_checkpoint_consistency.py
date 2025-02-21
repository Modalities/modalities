import glob
import json
import re
from pathlib import Path


def _get_checkpoint_file_name_without_eid(checkpoint_file_name: str) -> str:
    # Remove the experiment id from the checkpoint file name
    return re.sub(r"^eid_\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}_[a-f0-9]+-", "", checkpoint_file_name)


def test_checkpoint_files_exist(checkpoint_folder_path: list[Path], expected_checkpoint_names: list[str]):
    # Check if all the checkpoint files exist and have the correct names
    checkpoint_paths = glob.glob(str(checkpoint_folder_path / "**/*.bin"), recursive=True)

    assert len(checkpoint_paths) == 6, "ERROR! Expected 6 checkpoint files."

    for checkpoint_path in checkpoint_paths:
        checkpoint_file_name = Path(checkpoint_path).name
        cleaned_checkpoint_file_name = _get_checkpoint_file_name_without_eid(checkpoint_file_name)

        assert (
            cleaned_checkpoint_file_name in expected_checkpoint_names
        ), f"ERROR! {checkpoint_file_name} is not a valid checkpoint file name."


def check_last_checkpoint_info_correctness(checkpoint_folder_path: Path, expected_last_checkpoint_names: list[str]):
    # Check if the last checkpoint info files reference the correct checkpoint files

    checkpoint_info_paths = glob.glob(str(checkpoint_folder_path / "**/*.json"), recursive=True)

    assert len(checkpoint_info_paths) == 2, "ERROR! Expected 2 checkpoint info files."

    assert len(set(checkpoint_info_paths)) == len(
        checkpoint_info_paths
    ), "ERROR! Duplicate checkpoint info files found."

    for checkpoint_info_path in checkpoint_info_paths:
        with open(checkpoint_info_path, "r") as f:
            checkpoint_info = json.load(f)
        model_checkpoint_path = Path(checkpoint_info["model_checkpoint_path"])
        optimizer_checkpoint_path = Path(checkpoint_info["optimizer_checkpoint_path"])
        assert model_checkpoint_path.exists(), f"ERROR! {model_checkpoint_path} does not exist."
        assert optimizer_checkpoint_path.exists(), f"ERROR! {optimizer_checkpoint_path} does not exist."

        cleaned_model_checkpoint_file_name = _get_checkpoint_file_name_without_eid(model_checkpoint_path.name)
        cleaned_optimizer_checkpoint_file_name = _get_checkpoint_file_name_without_eid(optimizer_checkpoint_path.name)

        assert cleaned_model_checkpoint_file_name in expected_last_checkpoint_names
        assert cleaned_optimizer_checkpoint_file_name in expected_last_checkpoint_names


if __name__ == "__main__":
    checkpoint_folder_path = Path("../data/checkpoints")

    expected_checkpoint_names = [
        # pretrain checkpoint
        "model-seen_steps_11-seen_tokens_45056-target_steps_20-target_tokens_81920.bin",
        "optimizer-seen_steps_11-seen_tokens_45056-target_steps_20-target_tokens_81920.bin",
        # warmstart checkpoints
        "model-seen_steps_15-seen_tokens_61440-target_steps_20-target_tokens_81920.bin",
        "optimizer-seen_steps_15-seen_tokens_61440-target_steps_20-target_tokens_81920.bin",
        "model-seen_steps_20-seen_tokens_81920-target_steps_20-target_tokens_81920.bin",
        "optimizer-seen_steps_20-seen_tokens_81920-target_steps_20-target_tokens_81920.bin",
    ]

    expected_last_checkpoint_names = [
        # pretrain checkpoint
        "model-seen_steps_11-seen_tokens_45056-target_steps_20-target_tokens_81920.bin",
        "optimizer-seen_steps_11-seen_tokens_45056-target_steps_20-target_tokens_81920.bin",
        # warmstart checkpoints
        "model-seen_steps_20-seen_tokens_81920-target_steps_20-target_tokens_81920.bin",
        "optimizer-seen_steps_20-seen_tokens_81920-target_steps_20-target_tokens_81920.bin",
    ]

    test_checkpoint_files_exist(checkpoint_folder_path, expected_checkpoint_names)
    check_last_checkpoint_info_correctness(checkpoint_folder_path, expected_last_checkpoint_names)
