from pathlib import Path

import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_execution import CheckpointSavingExecutionABC
from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class TorchCheckpointSaving(CheckpointSavingExecutionABC):
    CHECKPOINT_STRUCTURE = "eid_{experiment_id}-{entity}-num_samples_{num_samples}.bin"

    def __init__(
        self,
        checkpoint_path: Path,
        experiment_id: str,
    ):
        self.checkpoint_path = checkpoint_path
        self.experiment_id = experiment_id

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        global_train_sample_id: int,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        raise NotImplementedError  # TODO Future work

    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, global_train_sample_id: int):
        raise NotImplementedError  # TODO Future work

    def _delete_checkpoint(self, global_train_sample_id: int):
        raise NotImplementedError  # TODO Future work
