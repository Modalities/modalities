import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    FullOptimStateDictConfig,
)
from pathlib import Path

from src.llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
import pytest


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

    checkpointing = FSDPToDiscCheckpointing(
        checkpoint_path=d, checkpointing_rank=0, experiment_id=str(1), global_rank=0, model_wrapping_fn=dummy_method
    )
    files_paths_to_delete = checkpointing._get_paths_to_delete(batch_id=100)
    assert len(files_paths_to_delete) != 0


def test_delete_checkpoint(tmpdir):
    experiment_id = "2023-11-17-01:35:09"
    directory = Path(tmpdir)

    optimizer_path = directory / f"{experiment_id}-optimizer-101.bin"
    optimizer_path.write_text(CONTENT)

    model_path = directory / f"{experiment_id}-model-101.bin"
    model_path.write_text(CONTENT)

    # "<experiment_id>-<enitity>-<step>.bin"
    checkpointing = FSDPToDiscCheckpointing(
        checkpoint_path=directory,
        checkpointing_rank=0,
        experiment_id=experiment_id,
        global_rank=0,
        model_wrapping_fn=dummy_method,
    )
    checkpointing._delete_checkpoint(batch_id=100)
    assert is_empty_directory(directory.__str__())
