import json
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
from modalities.utils.logging import get_logger

import torch.distributed.checkpoint as dcp
from modalities.checkpointing.fsdp.app_state import AppState

import os
from typing import Union
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict



class CheckpointingEntityType(Enum):
    """
    Enum class representing the types of entities that can be checkpointed.

    Attributes:
        MODEL (str): Represents the model entity.
        OPTIMIZER (str): Represents the optimizer entity.
        DISTRIBUTED (str): Represents the distributed checkpoint (dcp) 
                           entity that comprises both model and optimizer.
    """

    MODEL = "model"
    OPTIMIZER = "optimizer"
    DISTRIBUTED = "distributed"


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
        entity_type: CheckpointingEntityType,
        num_seen_steps: int = None,
        num_seen_tokens: int = None,
        num_target_steps: int = None,
        num_target_tokens: int = None,
    ) -> Path:
        if entity_type == CheckpointingEntityType.DISTRIBUTED:
            full_path = Path(self.checkpoint_path, experiment_id, 'distributed_checkpoint')
        else:
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

    @classmethod
    def _dcp_to_torch_save(cls, key: str, dcp_checkpoint_dir: Union[str, os.PathLike], torch_save_path: Union[str, os.PathLike]):
        """
        This function is almost the same as dcp_to_torch_save
        https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/checkpoint/format_utils.py#L196

        The distributed checkpoint contains both the model and the optimizer. 
        Instead of converting both together like the original function, 
        only the part specified by the input argument key = 'model' or 'optim' 
        is converted to a self-contained checkpoint.

        Args:
            key: 'model' or 'optim'
            dcp_checkpoint_dir: directory that contains the distributed checkpoint files (__<rank>_0.distcp)
            torch_save_path: path for the converted, self-contained checkpoint
        """
        sd: STATE_DICT_TYPE = {}
        _load_state_dict(
            sd,
            storage_reader=FileSystemReader(dcp_checkpoint_dir),
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
        torch.save(sd["app"][key], torch_save_path)

    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, training_progress: TrainingProgress):
        get_logger().info("Gathering model and optimizer checkpoint...")

        # Note: the experiment_id is created independently on each rank using a timestamp,
        #       which means that they are not necessarily the same.
        #       The following broadcast ensures that all ranks have the very same experiment_id,
        #       which is needed for distributed checkpointing (DCP).
        # TODO: this does not work, problem still needs to be solved.
        experiment_id = self.experiment_id
        to_broadcast = [experiment_id]
        dist.broadcast_object_list(to_broadcast, src=0)
        self.experiment_id = experiment_id

        # save distributed checkpoint (DCP)
        distributed_checkpoint_path = self._get_checkpointing_path(
            experiment_id=self.experiment_id,
            entity_type=CheckpointingEntityType.DISTRIBUTED,
        )

        distributed_checkpoint_path.mkdir(parents=True, exist_ok=True)
        get_logger().info(f"Saving distributed model checkpoint to {distributed_checkpoint_path}...")
        state_dict = { "app": AppState(model, optimizer) }
        dcp.save(state_dict, checkpoint_id=distributed_checkpoint_path)
        get_logger().info("Distributed checkpoint saved.")

        if self.global_rank == 0:
            # save model
            model_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                entity_type=CheckpointingEntityType.MODEL,
                num_seen_steps=training_progress.num_seen_steps_total,
                num_seen_tokens=training_progress.num_seen_tokens_total,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
            )
            checkpoint_folder_path = model_checkpoint_path.parent

            get_logger().info(f"Saving self-contained model checkpoint to {model_checkpoint_path}...")
            self._dcp_to_torch_save('model', distributed_checkpoint_path, model_checkpoint_path)
            get_logger().info("Model checkpoint saved.")


            # save optimizer
            optimizer_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                entity_type=CheckpointingEntityType.OPTIMIZER,
                num_seen_steps=training_progress.num_seen_steps_total,
                num_seen_tokens=training_progress.num_seen_tokens_total,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
            )
            get_logger().info(f"Saving self-contained optimizer checkpoint to {optimizer_checkpoint_path}...")
            self._dcp_to_torch_save('optim', distributed_checkpoint_path, optimizer_checkpoint_path)
            get_logger().info("Optimizer checkpoint saved.")

            # save checkpoint info
            checkpoint_info = {
                "model_checkpoint_path": str(Path.absolute(model_checkpoint_path)),
                "optimizer_checkpoint_path": str(Path.absolute(optimizer_checkpoint_path)),
            }
            get_logger().info(f"Saving checkpoint info {checkpoint_info} to {checkpoint_folder_path}...")
            last_checkpoint_info_file_path = checkpoint_folder_path / "last_checkpoint_info.json"
            with open(last_checkpoint_info_file_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_info, f)
            get_logger().info("Checkpoint info saved.")

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
            # TODO: Why the imported CheckpointEntityType instead of CheckpointingEntityType from this module? Can they be merged?
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
