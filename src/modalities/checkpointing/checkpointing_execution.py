from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.optim import Optimizer

from modalities.checkpointing.checkpointing_instruction import CheckpointingInstruction
from modalities.exceptions import CheckpointingError
from modalities.running_env.env_utils import MixedPrecisionSettings




class CheckpointingEntityType(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"


class CheckpointingExecutionIF(ABC):
    @abstractmethod
    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_checkpoint(
        self,
        optimizer: Optimizer,
        wrapped_model: nn.Module,
        file_path: Path,
    ) -> Optimizer:
        raise NotImplementedError

    @abstractmethod
    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        global_train_sample_id: int,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        raise NotImplementedError


class CheckpointingExecution(CheckpointingExecutionIF):
    @abstractmethod
    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, global_train_sample_id: int):
        raise NotImplementedError

    @abstractmethod
    def _delete_checkpoint(self, global_train_sample_id: int):
        raise NotImplementedError

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        global_train_sample_id: int,
        model: FSDP,
        optimizer: Optimizer,
    ):
        if checkpointing_instruction.save_current:
            self._save_checkpoint(model=model, optimizer=optimizer, global_train_sample_id=global_train_sample_id)

        for global_train_sample_id in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(global_train_sample_id=global_train_sample_id)


class FSDPToDiscCheckpointing(CheckpointingExecution):
    CHECKPOINT_STRUCTURE = "eid_{experiment_id}-{entity}-num_samples_{num_samples}.bin"

    def __init__(
        self,
        checkpoint_path: Path,
        experiment_id: str,
        global_rank: int,
        block_names: List[str],
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
    ):
        """
        Implementation of checkpointing to disc via FSDP

        Args:
            checkpoint_path (Path): folder path to the checkpoint
            experiment_id (str): ID of the experiment
            global_rank (int): global rank within the current process group
        """
        self.checkpoint_path = checkpoint_path
        self.global_rank = global_rank
        self.experiment_id = experiment_id
        self.block_names = block_names
        self.mixed_precision_settings = mixed_precision_settings
        self.sharding_strategy = sharding_strategy

    def _get_checkpointing_path(
        self,
        experiment_id: str,
        global_train_sample_id: int,
        entity_type: CheckpointingEntityType,
    ) -> Path:
        entity_file_name = self.CHECKPOINT_STRUCTURE.format(
            experiment_id=experiment_id, entity=entity_type.value, num_samples=str(global_train_sample_id + 1)
        )

        full_path = Path(self.checkpoint_path, experiment_id, entity_file_name)
        return full_path

    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, global_train_sample_id: int):
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
            optimizer_state = optimizer.state_dict()  # this gets the optimizer state dict object for each rank
            optim_state_dict = FSDP.optim_state_dict(
                model=model, optim=optimizer, optim_state_dict=optimizer_state
            )  # all the state dicts of the different ranks are synchronized

        if self.global_rank == 0:
            # save model
            model_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                global_train_sample_id=global_train_sample_id,
                entity_type=CheckpointingEntityType.MODEL,
            )

            model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_state, model_checkpoint_path)

            # save optimizer
            optimize_checkpoint_path = self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                global_train_sample_id=global_train_sample_id,
                entity_type=CheckpointingEntityType.OPTIMIZER,
            )
            torch.save(optim_state_dict, optimize_checkpoint_path)
        # we need this barrier here, such that all processes exit this function at the same time
        # Since we run throughput measurements in the trainer, the non-checkpointing ranks would already
        # trigger the time measurement in the trainer and would then wait for the checkpointing rank,
        # leading to wrong throughput measurements.
        dist.barrier()

    def _get_paths_to_delete(self, global_train_sample_id: int) -> List[Path]:
        return [
            self._get_checkpointing_path(
                experiment_id=self.experiment_id, entity_type=entity_type, global_train_sample_id=global_train_sample_id
            )
            for entity_type in CheckpointingEntityType
        ]

    def _delete_checkpoint(self, global_train_sample_id: int):
        if self.global_rank != 0:
            return

        files_paths_to_delete = self._get_paths_to_delete(global_train_sample_id=global_train_sample_id)
        for full_path in files_paths_to_delete:
            if full_path.exists():
                # unlink removes the file
                full_path.unlink()
            else:
                raise CheckpointingError(f"Checkpoint {full_path} could not be removed. It does not exist!")

    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
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
        if self.global_rank == 0:
            # load model on rank 0 into CPU RAM
            model_state = torch.load(file_path)
            model.load_state_dict(model_state)

        # TODO nasty workaround to prevent circular imports
        from modalities.models.model_factory import ModelFactory

        fsdp_model = ModelFactory.get_fsdp_wrapped_model(
            model=model,
            sync_module_states=True,
            block_names=self.block_names,
            mixed_precision_settings=self.mixed_precision_settings,
            sharding_strategy=self.sharding_strategy,
        )
        return fsdp_model

    def load_optimizer_checkpoint(self, optimizer: Optimizer, wrapped_model: FSDP, file_path: Path) -> Optimizer:
        # load optimizer
        full_optimizer_state_dict = None
        if self.global_rank == 0:
            # load full optimizer state dict to rank 0 (CPU RAM)
            full_optimizer_state_dict = torch.load(file_path)

        # distribute the optimizer state dict from rank 0 to all the other ranks
        sharded_optimizer_state_dict = FSDP.scatter_full_optim_state_dict(
            full_optim_state_dict=full_optimizer_state_dict, model=wrapped_model, group=None
        )
        optimizer.load_state_dict(sharded_optimizer_state_dict)

        return optimizer
