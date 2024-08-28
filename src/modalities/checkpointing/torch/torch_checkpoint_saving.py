from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_execution import CheckpointEntityType, CheckpointSavingExecutionABC
from modalities.exceptions import CheckpointingError


class TorchCheckpointSaving(CheckpointSavingExecutionABC):
    def __init__(
        self,
        checkpoint_path: Path,
        experiment_id: str,
        submodule: str = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.experiment_id = experiment_id
        self.submodule = submodule

    def _save_checkpoint(self, model: nn.Module, optimizer: Optimizer, train_step_id: int):
        if self.submodule:
            model_state = getattr(model, self.submodule).state_dict()
        else:
            model_state = model.state_dict()
        model_checkpoint_path = self._get_checkpointing_path(
            experiment_id=self.experiment_id,
            train_step_id=train_step_id,
            entity_type=CheckpointEntityType.MODEL,
        )
        model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state, model_checkpoint_path)

        # save optimizer
        optimizer_state = optimizer.state_dict()
        optimize_checkpoint_path = self._get_checkpointing_path(
            experiment_id=self.experiment_id,
            train_step_id=train_step_id,
            entity_type=CheckpointEntityType.OPTIMIZER,
        )
        torch.save(optimizer_state, optimize_checkpoint_path)

    def _delete_checkpoint(self, train_step_id: int):
        files_paths_to_delete = self._get_paths_to_delete(train_step_id=train_step_id)
        for full_path in files_paths_to_delete:
            if full_path.exists():
                # unlink removes the file
                full_path.unlink()
            else:
                raise CheckpointingError(f"Checkpoint {full_path} could not be removed. It does not exist!")
