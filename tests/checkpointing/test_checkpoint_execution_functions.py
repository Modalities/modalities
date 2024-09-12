from pathlib import Path

import pytest
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import FSDPCheckpointSaving
from modalities.training.training_progress import TrainingProgress


@pytest.mark.skip
def dummy_method(module: nn.Module, flag: bool) -> FSDP:
    raise NotImplementedError


@pytest.mark.skip
def is_empty_directory(folder_path: str) -> bool:
    path = Path(folder_path)
    return not any(path.iterdir())


CONTENT = "model"


def test_get_paths_to_delete(tmp_path):  # pytest temp path
    checkpointing = FSDPCheckpointSaving(
        checkpoint_path=tmp_path,
        experiment_id=str(1),
        global_rank=0,
    )
    trining_progress = TrainingProgress(
        num_seen_tokens_current_run=5, num_seen_steps_current_run=10, num_target_tokens=40, num_target_steps=20
    )

    files_paths_to_delete = checkpointing._get_paths_to_delete(training_progress=trining_progress)
    assert len(files_paths_to_delete) == 2


def test_delete_checkpoint(tmpdir):
    experiment_id = "2022-05-07__14-31-22"
    training_progress = TrainingProgress(
        num_seen_tokens_current_run=5, num_seen_steps_current_run=10, num_target_tokens=40, num_target_steps=20
    )
    directory = Path(tmpdir)

    (directory / experiment_id).mkdir(exist_ok=True)
    optimizer_file_name = (
        f"eid_{experiment_id}-optimizer-seen_steps_{training_progress.num_seen_steps_total}"
        f"-seen_tokens_{training_progress.num_seen_tokens_total}"
        f"-target_steps_{training_progress.num_target_steps}"
        f"-target_tokens_{training_progress.num_target_tokens}.bin"
    )
    optimizer_path = directory / experiment_id / optimizer_file_name
    optimizer_path.write_text(CONTENT)

    model_file_name = (
        f"eid_{experiment_id}-model-seen_steps_{training_progress.num_seen_steps_total}"
        f"-seen_tokens_{training_progress.num_seen_tokens_total}"
        f"-target_steps_{training_progress.num_target_steps}"
        f"-target_tokens_{training_progress.num_target_tokens}.bin"
    )
    model_path = directory / experiment_id / model_file_name
    model_path.write_text(CONTENT)

    checkpoint_saving = FSDPCheckpointSaving(
        checkpoint_path=directory,
        experiment_id=experiment_id,
        global_rank=0,
    )
    checkpoint_saving._delete_checkpoint(training_progress=training_progress)
    assert is_empty_directory((directory / experiment_id).__str__())
