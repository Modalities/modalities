from enum import Enum
from typing import Callable, Tuple, List
from llm_gym.checkpointing.checkpointing import (
    CheckpointingExecutionIF,
    CheckpointingInstruction,
)
from llm_gym.exceptions import CheckpointingError
from llm_gym.models.model import NNModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    FullOptimStateDictConfig,
)
import torch
import os
from torch.optim import Optimizer
import torch.distributed as dist
import torch.nn as nn
import glob


class CheckpointingEntityType(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"


import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from llm_gym.checkpointing.checkpointing import CheckpointingExecutionIF, CheckpointingInstruction
from llm_gym.exceptions import CheckpointingError


class FSDPToDiscCheckpointing(CheckpointingExecutionIF):
    def __init__(
        self,
        checkpoint_path: str,
        experiment_id: str,
        global_rank: int,
        checkpointing_rank: int,
        model_wrapping_fn: Callable[[nn.Module, bool], FSDP],
    ):
        self.checkpoint_path = checkpoint_path
        self.global_rank = global_rank
        self.checkpointing_rank = checkpointing_rank
        self.model_wrapping_fn = model_wrapping_fn
        self.experiment_id = experiment_id
        self.checkpoint_structure = f"<experiment_id>-<enitity>-<step>.bin"

    def run_checkpoint_instructions(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        global_train_batch_id: int,
        model: FSDP,
        optimizer: Optimizer,
    ):
        if checkpointing_instruction.save_current:
            self._save_checkpoint(model=model, optimizer=optimizer, global_train_batch_id=global_train_batch_id)

        for batch_id in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(batch_id=batch_id)

    def _get_checkpointing_path(
        self,
        experiment_id: str,
        global_train_batch_id: int,
        entity_type: CheckpointingEntityType,
    ):
        entity_file_name = (
            self.checkpoint_structure.replace("<experiment_id>", experiment_id)
            .replace("<enitity>", entity_type.value)
            .replace("<step>", str(global_train_batch_id + 1))
        )
        full_path = os.path.join(self.checkpoint_path, entity_file_name)
        return full_path

    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, global_train_batch_id: int):
        # saving the model via FULL_STATE_DICT and checkpoint via FULL_OPTIM_STATE_DICT
        # TODO Need to check if LR schedulers also need checkpointing
        model_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=model_save_policy,
            optim_state_dict_config=optim_save_policy,
        ):
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict() # this gets the optimizer state dict object for each rank
            optim_state_dict = FSDP.optim_state_dict(model=model, optim=optimizer, optim_state_dict=optimizer_state) # all the state dicts of the different ranks are synchronized 

        if self.checkpointing_rank == self.global_rank:
            # save model
            model_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                global_train_batch_id=global_train_batch_id,
                entity_type=CheckpointingEntityType.MODEL,
            )
            torch.save(model_state, model_checkpoint_path)

            # save optimizer
            optimize_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                global_train_batch_id=global_train_batch_id,
                entity_type=CheckpointingEntityType.OPTIMIZER,
            )
            torch.save(optim_state_dict, optimize_checkpoint_path)


    def _get_paths_to_delete(self, batch_id: int)-> List[str]:
        entity_file_name_regex = self.checkpoint_structure.replace("<enitity>", "*").replace(
            "<step>", str(batch_id + 1)
        )
        full_path_regex = os.path.join(self.checkpoint_path, entity_file_name_regex)
        files_paths_to_delete = glob.glob(full_path_regex)
        return files_paths_to_delete


    def _delete_checkpoint(self, batch_id: int):
        if self.global_rank != 0:
            return
        # TODO we need more logic to also delete optimizers and lr schedulers
        entity_file_name = self.checkpoint_structure.replace("<enitity>", "model").replace("<step>", str(batch_id + 1))
        full_path = os.path.join(self.checkpoint_path, entity_file_name)
        if os.path.exists(full_path):
            os.remove(full_path)
        else:
            raise CheckpointingError(f"Checkpoint {full_path} could not be removed. It does not exist!")

        files_paths_to_delete = self._get_paths_to_delete(batch_id)
        for full_path in files_paths_to_delete:
            if os.path.exists(full_path):
                os.remove(full_path)
            else:
                raise CheckpointingError(f"Checkpoint {full_path} could not be removed. It does not exist!")

    def load_model_checkpoint(self, model: nn.Module, experiment_id: str, global_train_batch_id: int) -> nn.Module:
        # Loads the checkpoint as full state dicts into the model and optimizer on rank 0.
        # NOTE: The model and optimizer need to be sharded after calling this function!

        # model_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # with FSDP.state_dict_type(
        #     module=model,
        #     state_dict_type=StateDictType.FULL_STATE_DICT,
        #     state_dict_config=model_save_policy,
        #     optim_state_dict_config=optim_save_policy,
        # ):
        # we only load the model and optimizer on a single rank. The calling function must then
        # distribute the optimizer state and model parmeters to the other ranks.

        # load model
        if self.global_rank == self.checkpointing_rank:
            # load model on rank 0 into CPU RAM
            model_checkpoint_path = self._get_checkpointing_path(
                experiment_id=experiment_id,
                global_train_batch_id=global_train_batch_id,
                entity_type=CheckpointingEntityType.MODEL,
            )
            model_state = torch.load(model_checkpoint_path)
            model.load_state_dict(model_state)
        fsdp_model = self.model_wrapping_fn(model=model, sync_module_states=True)
        return fsdp_model

    def load_optimizer_checkpoint(
        self, optimizer: Optimizer, model: FSDP, experiment_id: str, global_train_batch_id: int
    ) -> Optimizer:
        # load optimizer
        full_optimizer_state_dict = None
        if self.global_rank == self.checkpointing_rank:
            # load full optimizer state dict to rank 0 (CPU RAM)
            optimizer_checkpoint_path = self._get_checkpointing_path(
                experiment_id=experiment_id,
                global_train_batch_id=global_train_batch_id,
                entity_type=CheckpointingEntityType.OPTIMIZER,
            )
            full_optimizer_state_dict = torch.load(optimizer_checkpoint_path)

        # distribute the optimizer state dict from rank 0 to all the other ranks
        sharded_optimizer_state_dict = FSDP.scatter_full_optim_state_dict(
            full_optim_state_dict=full_optimizer_state_dict, model=model, group=None
        )
        optimizer.load_state_dict(sharded_optimizer_state_dict)

        return optimizer
