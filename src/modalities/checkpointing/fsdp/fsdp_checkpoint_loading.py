from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.running_env.env_utils import MixedPrecisionSettings


class FSDPCheckpointLoading(CheckpointLoadingIF):
    def __init__(
        self,
        global_rank: int,
        block_names: List[str],
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
    ):
        self.global_rank = global_rank
        self.block_names = block_names
        self.mixed_precision_settings = mixed_precision_settings
        self.sharding_strategy = sharding_strategy

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
