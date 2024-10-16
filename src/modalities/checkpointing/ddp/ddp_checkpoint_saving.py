from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_execution import CheckpointEntityType, CheckpointSavingExecutionABC
from modalities.exceptions import CheckpointingError
from modalities.training.training_progress import TrainingProgress


class DDPCheckpointSaving(CheckpointSavingExecutionABC):
    """DDPCheckpointSaving class for saving checkpoints of DDP models and optimizers."""

    CHECKPOINT_STRUCTURE = (
        "eid_{experiment_id}-{entity}-seen_steps_{num_seen_steps}-seen_tokens_{num_seen_tokens}"
        "-target_steps_{num_target_steps}-target_tokens_{num_target_tokens}.bin"
    )

    def __init__(
        self,
        checkpoint_path: Path,
        experiment_id: str,
        global_rank: int,
        submodule: str = None,
        lora_submodule: str = None,
    ):
        """
        Initializes the DDPCheckpointSaving class.

        Args:
            checkpoint_path (Path): folder path to the checkpoint
            experiment_id (str): ID of the experiment
            global_rank (int): global rank within the current process group
            submodule (str): if specified, only a submodule of the model will be saved
            lora_submodule (str): if specified, only the lora weights of the submodule will be saved

         Returns:
            None
        """
        self.checkpoint_path = checkpoint_path
        self.experiment_id = experiment_id
        self.global_rank = global_rank
        self.submodule = submodule
        self.lora_submodule = lora_submodule

    def _save_checkpoint(self, model: nn.Module, optimizer: Optimizer, training_progress: TrainingProgress):
        if self.global_rank == 0:
            model = model.module

            if self.submodule:
                model_state = getattr(model, self.submodule).state_dict()
            else:
                model_state = model.state_dict()
            model_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                num_seen_steps=training_progress.num_seen_steps_total,
                num_seen_tokens=training_progress.num_seen_tokens_total,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
                entity_type=CheckpointEntityType.MODEL,
            )
            model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_state, model_checkpoint_path)

            # save optimizer
            optimizer_state = optimizer.state_dict()
            # save optimizer
            optimize_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                num_seen_steps=training_progress.num_seen_steps_total,
                num_seen_tokens=training_progress.num_seen_tokens_total,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
                entity_type=CheckpointEntityType.OPTIMIZER,
            )
            torch.save(optimizer_state, optimize_checkpoint_path)
            if self.lora_submodule:
                lora_model = getattr(model, self.lora_submodule)
                lora_model.save_pretrained(model_checkpoint_path.parents[0])
        # we need this barrier here, such that all processes exit this function at the same time
        # Since we run throughput measurements in the trainer, the non-checkpointing ranks would already
        # trigger the time measurement in the trainer and would then wait for the checkpointing rank,
        # leading to wrong throughput measurements.
        dist.barrier()

    def _delete_checkpoint(self, training_progress: TrainingProgress):
        if self.global_rank == 0:
            files_paths_to_delete = self._get_paths_to_delete(training_progress=training_progress)
            for full_path in files_paths_to_delete:
                if full_path.exists():
                    # unlink removes the file
                    full_path.unlink()
                else:
                    raise CheckpointingError(f"Checkpoint {full_path} could not be removed. It does not exist!")
