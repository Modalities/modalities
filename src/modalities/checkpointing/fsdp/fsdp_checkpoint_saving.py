from enum import Enum
from pathlib import Path
from typing import Callable, List

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving import CheckpointEntityType
from modalities.checkpointing.checkpoint_saving_execution import CheckpointSavingExecutionABC
from modalities.exceptions import CheckpointingError


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

    CHECKPOINT_STRUCTURE = "eid_{experiment_id}-{entity}-num_steps_{num_train_steps}-num_tokens_{num_tokens}.bin"

    def __init__(
        self,
        checkpoint_path: Path,
        experiment_id: str,
        global_rank: int,
        get_num_tokens_from_num_steps_callable: Callable[[int], int],
    ):
        """
        Initializes the FSDPCheckpointSaving class.

        Args:
            checkpoint_path (Path): folder path to the checkpoint
            experiment_id (str): ID of the experiment
            global_rank (int): global rank within the current process group
            get_num_tokens_from_num_steps_callable (Callable[[int], int]): callable to get the number
                of tokens for a given number of train steps

         Returns:
            None
        """
        self.checkpoint_path = checkpoint_path
        self.global_rank = global_rank
        self.experiment_id = experiment_id
        self.get_num_tokens_from_num_steps_callable = get_num_tokens_from_num_steps_callable

    def _get_checkpointing_path(
        self,
        experiment_id: str,
        num_train_steps_done: int,
        entity_type: CheckpointingEntityType,
    ) -> Path:
        """
        Returns the path for saving the checkpoint file.

        Args:
            experiment_id (str): The ID of the experiment.
            num_train_steps_done (int): The number of training steps completed.
            entity_type (CheckpointingEntityType): The type of entity (model or optimizer) being checkpointed.

        Returns:
            Path: The path for saving the checkpoint file.
        """
        num_tokens = self.get_num_tokens_from_num_steps_callable(num_train_steps_done)
        entity_file_name = self.CHECKPOINT_STRUCTURE.format(
            experiment_id=experiment_id,
            entity=entity_type.value,
            num_train_steps=str(num_train_steps_done),
            num_tokens=str(num_tokens),
        )

        full_path = Path(self.checkpoint_path, experiment_id, entity_file_name)
        return full_path

    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, num_train_steps_done: int):
        """
        Saves the model and optimizer checkpoints.

        Args:
            model (FSDP): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.
            num_train_steps_done (int): The number of training steps completed.

        Returns:
            None
        """
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
                num_train_steps_done=num_train_steps_done,
                entity_type=CheckpointingEntityType.MODEL,
            )

            model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_state, model_checkpoint_path)

            # save optimizer
            optimize_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                num_train_steps_done=num_train_steps_done,
                entity_type=CheckpointingEntityType.OPTIMIZER,
            )
            torch.save(optim_state_dict, optimize_checkpoint_path)
        # we need this barrier here, such that all processes exit this function at the same time
        # Since we run throughput measurements in the trainer, the non-checkpointing ranks would already
        # trigger the time measurement in the trainer and would then wait for the checkpointing rank,
        # leading to wrong throughput measurements.
        dist.barrier()

    def _get_paths_to_delete(self, num_train_steps_done: int) -> List[Path]:
        """
        Returns a list of paths to delete for checkpointing.

        Args:
            num_train_steps_done (int): The number of training steps completed.

        Returns:
            List[Path]: A list of paths to delete.
        """
        return [
            self._get_checkpointing_path(
                experiment_id=self.experiment_id, entity_type=entity_type, num_train_steps_done=num_train_steps_done
            )
            for entity_type in CheckpointEntityType
        ]

    def _delete_checkpoint(self, num_train_steps_done: int):
        """
        Deletes the checkpoint file.

        Args:
            num_train_steps_done (int): The number of train steps completed.

        Raises:
            CheckpointingError: If the checkpoint file could not be removed because it does not exist.

        Returns:
            None
        """
        if self.global_rank != 0:
            return

        files_paths_to_delete = self._get_paths_to_delete(num_train_steps_done=num_train_steps_done)
        for full_path in files_paths_to_delete:
            if full_path.exists():
                # unlink removes the file
                full_path.unlink()
            else:
                raise CheckpointingError(f"Checkpoint {full_path} could not be removed. It does not exist!")
