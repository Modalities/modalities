from llm_gym.checkpointing.checkpointing import (
    CheckpointingExecutionIF,
    CheckpointingInstruction,
)
from llm_gym.exceptions import CheckpointingError
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
import torch
import os


class FSDPToDiscCheckpointing(CheckpointingExecutionIF):
    def __init__(
        self,
        checkpoint_path: str,
        experiment_id: str,
        global_rank: int,
        checkpointing_rank: int,
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_structure = f"{experiment_id}-<enitity>-<step>.bin"
        self.global_rank = global_rank
        self.checkpointing_rank = checkpointing_rank

    def run_checkpoint_instructions(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        global_train_batch_id: int,
        model: FSDP,
    ):
        if checkpointing_instruction.save_current:
            self._save_checkpoint(model=model, global_train_batch_id=global_train_batch_id)

        for batch_id in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(batch_id=batch_id)

    def _save_checkpoint(self, model: FSDP, global_train_batch_id: int):
        # print(f"Saving model checkpoint for epoch {current_epoch}...\n")
        # TODO add optimizer checkpointing
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict
        # Need to check if LR schedulers also need checkpointing
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        if self.checkpointing_rank == self.global_rank:
            entity_file_name = self.checkpoint_structure.replace(
                "<enitity>", "model"
            ).replace("<step>", str(global_train_batch_id + 1))
            full_path = os.path.join(self.checkpoint_path, entity_file_name)
            torch.save(cpu_state, full_path)
            # print(f"----> Model checkpoint saved at {full_path} for epoch {current_epoch} by rank {self.global_rank}\n")

    def _delete_checkpoint(self, batch_id: int):
        if self.global_rank != 0:
            return 
        # TODO we need more logic to also delete optimizers and lr schedulers
        entity_file_name = self.checkpoint_structure.replace(
            "<enitity>", "model"
        ).replace("<step>", str(batch_id + 1))
        full_path = os.path.join(self.checkpoint_path, entity_file_name)
        if os.path.exists(full_path):
            os.remove(full_path)
        else:
            raise CheckpointingError(
                f"Checkpoint {full_path} could not be removed. It does not exist!"
            )
