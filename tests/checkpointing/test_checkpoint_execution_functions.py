from pathlib import Path

import pytest
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import FSDPCheckpointSaving
from modalities.utils.number_conversion import NumberConversion


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
        get_num_tokens_from_num_steps_callable=lambda _: 0,
    )
    files_paths_to_delete = checkpointing._get_paths_to_delete(num_train_steps_done=101)
    assert len(files_paths_to_delete) == 2


def test_delete_checkpoint(tmpdir):
    experiment_id = "2022-05-07__14-31-22"
    directory = Path(tmpdir)

    (directory / experiment_id).mkdir(exist_ok=True)

    optimizer_path = directory / experiment_id / f"eid_{experiment_id}-optimizer-num_steps_101-num_tokens_4848.bin"
    optimizer_path.write_text(CONTENT)

    model_path = directory / experiment_id / f"eid_{experiment_id}-model-num_steps_101-num_tokens_4848.bin"
    model_path.write_text(CONTENT)
    get_num_tokens_from_num_steps_callable = NumberConversion.get_num_tokens_from_num_steps_callable(
        num_ranks=2, local_micro_batch_size=4, sequence_length=6
    )
    checkpoint_saving = FSDPCheckpointSaving(
        checkpoint_path=directory,
        experiment_id=experiment_id,
        global_rank=0,
        get_num_tokens_from_num_steps_callable=get_num_tokens_from_num_steps_callable,
    )
    checkpoint_saving._delete_checkpoint(num_train_steps_done=101)
    assert is_empty_directory((directory / experiment_id).__str__())
