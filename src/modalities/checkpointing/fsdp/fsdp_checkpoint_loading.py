from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_loading import DistributedCheckpointLoadingIF, FSDP1CheckpointLoadingIF
from modalities.checkpointing.stateful.app_state import AppState
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.utils.logger_utils import get_logger


class FSDP1CheckpointLoading(FSDP1CheckpointLoadingIF):
    """FSDP1 checkpoint loading class."""

    def __init__(
        self,
        global_rank: int,
        block_names: list[str],
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
    ):
        """
        Initializes the FSDP1CheckpointLoading object.

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

    # Ensures reduced memory footprint and avoids side-effects
    @torch.no_grad()
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
        get_logger().info(f"Loading model checkpoint from {file_path} on rank {self.global_rank}...")
        # load model
        if self.global_rank == 0:
            # load model on rank 0 into CPU RAM
            model_state = torch.load(file_path)
            model.load_state_dict(model_state)

        # TODO nasty workaround to prevent circular imports
        from modalities.models.model_factory import ModelFactory

        fsdp_model = ModelFactory.get_fsdp1_wrapped_model(
            model=model,
            sync_module_states=True,
            block_names=self.block_names,
            mixed_precision_settings=self.mixed_precision_settings,
            sharding_strategy=self.sharding_strategy,
        )
        get_logger().info(f"Model checkpoint loaded on rank {self.global_rank}.")
        return fsdp_model

    def load_optimizer_checkpoint_(self, optimizer: Optimizer, model: FSDP, file_path: Path):
        """
        Loads the checkpoint as full state dict into the optimizer on rank 0 (in-place)

        Args:
            optimizer (Optimizer): The optimizer to load the checkpoint into (in-place).
            model (FSDP): The FSDP-wrapped model.
            file_path (Path): The path to the checkpoint file.
        """
        get_logger().info(f"Loading optimizer checkpoint from {file_path} on rank {self.global_rank}...")
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
        get_logger().info(f"Optimizer checkpoint loaded on rank {self.global_rank}.")


class DCPCheckpointLoading(DistributedCheckpointLoadingIF):
    """Distributed checkpoint loading interface for loading PyTorch models and optimizer checkpoints."""

    def __init__(self, global_rank: int):
        """
        Initializes the DCPCheckpointLoading object.

        Args:
            global_rank (int): The global rank of the process.

        Returns:
            None
        """
        self._global_rank = global_rank

    @torch.no_grad()
    def load_checkpoint_(self, app_state: AppState, checkpoint_dir_path: Path):
        """Loads the distributed checkpoint from the specified directory path.
        NOTE: The model in the app_state must be already FSDP-wrapped.

        Args:
            app_state (AppState): The application state with the model and optimizer.
            checkpoint_directory_path (Path): The directory path to the distributed checkpoint.
        """

        get_logger().info(f"Loading distributed checkpoint on rank {self._global_rank}.")
        dcp.load(
            state_dict={"app": app_state},
            checkpoint_id=checkpoint_dir_path,
        )
        get_logger().info(f"Distributed checkpoint loaded on rank {self._global_rank}.")
