from pathlib import Path

import pytest
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import FSDPCheckpointSaving


@pytest.mark.skip
def dummy_method(module: nn.Module, flag: bool) -> FSDP:
    raise NotImplementedError


@pytest.mark.skip
def is_empty_directory(folder_path: str) -> bool:
    path = Path(folder_path)
    return not any(path.iterdir())


CONTENT = "model"


def test_get_paths_to_delete(tmp_path):  # pytest temp path
    d = tmp_path / "folder"
    d.mkdir()
    p = d / "<experiment_id>-200-101.bin"
    p.write_text(CONTENT)

    checkpointing = FSDPCheckpointSaving(checkpoint_path=d, experiment_id=str(1), global_rank=0)
    files_paths_to_delete = checkpointing._get_paths_to_delete(train_step_id=100)
    assert len(files_paths_to_delete) != 0


def test_delete_checkpoint(tmpdir):
    experiment_id = "2022-05-07__14-31-22"
    directory = Path(tmpdir)

    (directory / experiment_id).mkdir(exist_ok=True)

    optimizer_path = directory / experiment_id / f"eid_{experiment_id}-optimizer-num_steps_101.bin"
    optimizer_path.write_text(CONTENT)

    model_path = directory / experiment_id / f"eid_{experiment_id}-model-num_steps_101.bin"
    model_path.write_text(CONTENT)

    checkpoint_saving = FSDPCheckpointSaving(checkpoint_path=directory, experiment_id=experiment_id, global_rank=0)
    checkpoint_saving._delete_checkpoint(train_step_id=100)
    assert is_empty_directory((directory / experiment_id).__str__())
