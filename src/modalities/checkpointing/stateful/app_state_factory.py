from pathlib import Path
from typing import Optional

import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import DCPCheckpointLoading
from modalities.checkpointing.stateful.app_state import AppState


class AppStateFactory:
    """Factory class to create AppState objects."""

    @staticmethod
    def get_raw_app_state(
        model: nn.Module, optimizer: Optimizer, lr_scheduler: Optional[LRScheduler] = None
    ) -> AppState:
        """Creates a new (non-checkpoint loaded) AppState object from an instanitated
        model, optimizer, and optional learning rate scheduler.

        Args:
            model (nn.Module): The model can be either a non-sharded model, FSDP1 or FSDP2 model.
            optimizer (Optimizer): The optimizer can be either a non-sharded optimizer, FSDP1 or FSDP2 optimizer.
            lr_scheduler (Optional[LRScheduler], optional): Lr scheduler used during training. Defaults to None.

        Returns:
            AppState: The AppState object.
        """
        app_state = AppState(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
        return app_state

    @staticmethod
    def get_dcp_checkpointed_app_state(
        raw_app_state: AppState,
        checkpoint_dir_path: Path,
    ) -> AppState:
        """Loads the checkpointed state dict into the raw AppState object (i.e., non-checkpoint loaded AppState).

        Args:
            raw_app_state (AppState): The raw AppState object.
            checkpoint_dir_path (Path): The path to the checkpoint directory.

        Raises:
            RuntimeError: Raises an error if the state dict has already been loaded.

        Returns:
            AppState: The AppState object with the loaded state dict.
        """
        if raw_app_state.is_loaded:
            raise RuntimeError(
                "Cannot call load_state_dict twice on the same AppState object. " "State dict has already been loaded."
            )
        cp_loading = DCPCheckpointLoading(global_rank=dist.get_rank())
        cp_loading.load_checkpoint_(app_state=raw_app_state, checkpoint_dir_path=checkpoint_dir_path)
        return raw_app_state
