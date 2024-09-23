from pathlib import Path

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.running_env.env_utils import MixedPrecisionSettings


class FSDPCheckpointLoading(CheckpointLoadingIF):
    """FSDP checkpoint loading class."""

    def __init__(
        self,
        global_rank: int,
        block_names: list[str],
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
    ):
        """
        Initializes the FSDPCheckpointLoading object.

        Args:
            global_rank (int): The global rank of the process.
            block_names (list[str]): The names of the blocks.
            mixed_precision_settings (MixedPrecisionSettings): The settings for mixed precision.
            sharding_strategy (ShardingStrategy): The sharding strategy.

        Returns:
            None
        """
        self.global_rank = global_rank
        self.block_names = block_names
        self.mixed_precision_settings = mixed_precision_settings
        self.sharding_strategy = sharding_strategy

    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        """
        Loads the checkpoint as full state dict into the model on rank 0.
        After loading the model to CPU RAM, the model is wrapped with FSDP and sharded
        across the ranks according to the sharding strategy.

        Args:
            model (nn.Module): The model to load the checkpoint into.
            file_path (Path): The path to the checkpoint file.

        Returns:
            nn.Module: The model wrapped with FSDP and sharded according to the sharding strategy.
        """

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

    def load_optimizer_checkpoint(self, optimizer: Optimizer, model: FSDP, file_path: Path) -> Optimizer:
        """
        Loads the checkpoint as full state dict into the optimizer on rank 0

        Args:
            optimizer (Optimizer): The optimizer to load the checkpoint into.
            model (FSDP): The FSDP-wrapped model.
            file_path (Path): The path to the checkpoint file.

        Returns:
            Optimizer: The optimizer with the loaded checkpoint.
        """
        # NOTE: model must be FSDP-wrapped model!
        # load optimizer
        full_optimizer_state_dict = None
        if self.global_rank == 0:
            # load full optimizer state dict to rank 0 (CPU RAM)
            full_optimizer_state_dict = torch.load(file_path)

        # distribute the optimizer state dict from rank 0 to all the other ranks
        sharded_optimizer_state_dict = FSDP.scatter_full_optim_state_dict(
            full_optim_state_dict=full_optimizer_state_dict, model=model, group=None
        )
        optimizer.load_state_dict(sharded_optimizer_state_dict)

        return optimizer
