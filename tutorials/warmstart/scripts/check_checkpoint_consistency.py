import glob
import os
import re
from pathlib import Path


def _get_checkpoint_file_name_without_eid(checkpoint_file_name: str) -> str:
    # Remove the experiment id from the checkpoint file name
    return re.sub(r"^eid_\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}_[a-f0-9]+-", "", checkpoint_file_name)


def test_checkpoint_files_exist(checkpoint_folder_path: list[Path], expected_checkpoint_names: list[str]):
    # Check if all the checkpoint files exist and have the correct names
    checkpoint_paths = glob.glob(str(checkpoint_folder_path / "**/*"), recursive=True)

    assert len(checkpoint_paths) == 17, "ERROR! Expected 6 checkpoint files."

    assert len([p for p in checkpoint_paths if p.endswith(".distcp")]), "ERROR! Expected 6 checkpoint files."


if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    os.chdir(current_file_path.parent)

    checkpoint_folder_path = Path("../data/checkpoints")

    expected_checkpoint_folder_names = [
        # pretrain checkpoint
        "seen_steps_11-seen_tokens_45056-target_steps_20-target_tokens_81920",
        # warmstart checkpoints
        "seen_steps_15-seen_tokens_61440-target_steps_20-target_tokens_81920",
        "seen_steps_20-seen_tokens_81920-target_steps_20-target_tokens_81920",
    ]

    test_checkpoint_files_exist(checkpoint_folder_path, expected_checkpoint_folder_names)
