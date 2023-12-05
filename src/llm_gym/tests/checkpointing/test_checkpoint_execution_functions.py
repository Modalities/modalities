from typing import List
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    FullOptimStateDictConfig,
)
from pathlib import Path

from src.llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing, CheckpointingEntityType
import pytest


@pytest.mark.skip
def dummy_method(
        module: nn.Module,
        flag: bool
) -> FSDP:
    raise NotImplementedError


@pytest.mark.skip
def is_empty_directory(
        folder_path: str
) -> bool:
    path = Path(folder_path)
    return not any(path.iterdir())


CONTENT = "model"


def test_get_paths_to_delete(
        tmp_path  # pytest temp path
):
    d = tmp_path / "folder"
    d.mkdir()
    p = d / '<experiment_id>-200-101.bin'
    p.write_text(CONTENT)

    checkpointing = FSDPToDiscCheckpointing(
        checkpoint_path=d.__str__(),
        checkpointing_rank=0,
        experiment_id=str(1),
        global_rank=0,
        model_wrapping_fn=dummy_method
    )
    files_paths_to_delete = checkpointing._get_paths_to_delete(batch_id=100)
    assert len(files_paths_to_delete) != 0


def test_delete_checkpoint(tmp_path):
    d = tmp_path / "folder"
    d.mkdir()
    p = d / '<experiment_id>-200-101.bin'
    p.write_text(CONTENT)
    p = d / '<experiment_id>-model-101.bin'
    p.write_text(CONTENT)
    # "<experiment_id>-<enitity>-<step>.bin"
    flagger = is_empty_directory(d.__str__())
    checkpointing = FSDPToDiscCheckpointing(checkpoint_path=d.__str__(), checkpointing_rank=0, experiment_id=str(1),
                                            global_rank=0,
                                            model_wrapping_fn=dummy_method)
    checkpointing._delete_checkpoint(batch_id=100)
    assert is_empty_directory(d.__str__()) == True
