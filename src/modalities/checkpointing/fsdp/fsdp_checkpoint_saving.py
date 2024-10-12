from enum import Enum
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving import CheckpointEntityType
from modalities.checkpointing.checkpoint_saving_execution import CheckpointSavingExecutionABC
from modalities.exceptions import CheckpointingError
from modalities.training.training_progress import TrainingProgress


class CheckpointingEntityType(Enum):
    """
    Enum class representing the types of entities that can be checkpointed.

    Attributes:
        MODEL (str): Represents the model entity.
        OPTIMIZER (str): Represents the optimizer entity.
    """

    MODEL = "model"
    OPTIMIZER = "optimizer"


class FSDPCheckpointSaving(CheckpointSavingExecutionABC):
    """FSDPCheckpointSaving class for saving checkpoints of FSDP models and optimizers."""

    CHECKPOINT_STRUCTURE = (
        "eid_{experiment_id}-{entity}-seen_steps_{num_seen_steps}-seen_tokens_{num_seen_tokens}"
        "-target_steps_{num_target_steps}-target_tokens_{num_target_tokens}.bin"
    )

    def __init__(
        self,
        checkpoint_path: Path,
        experiment_id: str,
        global_rank: int,
    ):
        """
        Initializes the FSDPCheckpointSaving class.

        Args:
            checkpoint_path (Path): folder path to the checkpoint
            experiment_id (str): ID of the experiment
            global_rank (int): global rank within the current process group

         Returns:
            None
        """
        self.checkpoint_path = checkpoint_path
        self.global_rank = global_rank
        self.experiment_id = experiment_id

    def _get_checkpointing_path(
        self,
        experiment_id: str,
        num_seen_steps: int,
        num_seen_tokens: int,
        num_target_steps: int,
        num_target_tokens: int,
        entity_type: CheckpointingEntityType,
    ) -> Path:
        entity_file_name = self.CHECKPOINT_STRUCTURE.format(
            experiment_id=experiment_id,
            entity=entity_type.value,
            num_seen_steps=str(num_seen_steps),
            num_seen_tokens=str(num_seen_tokens),
            num_target_steps=str(num_target_steps),
            num_target_tokens=str(num_target_tokens),
        )

        full_path = Path(self.checkpoint_path, experiment_id, entity_file_name)
        return full_path

    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, training_progress: TrainingProgress):
        # saving the model via FULL_STATE_DICT and checkpoint via FULL_OPTIM_STATE_DICT
        model_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=model_save_policy,
            optim_state_dict_config=optim_save_policy,
        ):
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()  # this gets the optimizer state dict object for each rank
            optim_state_dict = FSDP.optim_state_dict(
                model=model, optim=optimizer, optim_state_dict=optimizer_state
            )  # all the state dicts of the different ranks are synchronized

        if self.global_rank == 0:
            # save model
            model_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                num_seen_steps=training_progress.num_seen_steps_total,
                num_seen_tokens=training_progress.num_seen_tokens_total,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
                entity_type=CheckpointingEntityType.MODEL,
            )

            model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_state, model_checkpoint_path)

            # save optimizer
            optimize_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                num_seen_steps=training_progress.num_seen_steps_total,
                num_seen_tokens=training_progress.num_seen_tokens_total,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
                entity_type=CheckpointingEntityType.OPTIMIZER,
            )
            torch.save(optim_state_dict, optimize_checkpoint_path)
        # we need this barrier here, such that all processes exit this function at the same time
        # Since we run throughput measurements in the trainer, the non-checkpointing ranks would already
        # trigger the time measurement in the trainer and would then wait for the checkpointing rank,
        # leading to wrong throughput measurements.
        dist.barrier()

    def _get_paths_to_delete(self, training_progress: TrainingProgress) -> list[Path]:
        return [
            self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                entity_type=entity_type,
                num_seen_steps=training_progress.num_seen_steps_total,
                num_seen_tokens=training_progress.num_seen_tokens_total,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
            )
            for entity_type in CheckpointEntityType
        ]

    def _delete_checkpoint(self, training_progress: TrainingProgress):
        if self.global_rank != 0:
            return

        files_paths_to_delete = self._get_paths_to_delete(training_progress=training_progress)
        for full_path in files_paths_to_delete:
            if full_path.exists():
                # unlink removes the file
                full_path.unlink()
            else:
                raise CheckpointingError(f"Checkpoint {full_path} could not be removed. It does not exist!")
