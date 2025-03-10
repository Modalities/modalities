from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import DCPCheckpointLoading
from modalities.checkpointing.stateful.app_state import AppState


class AppStateFactory:
    def get_raw_app_state(self, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler) -> AppState:
        app_state = AppState(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
        return app_state

    def get_dcp_checkpointed_app_state(
        self,
        app_state: AppState,
        checkpoint_directory_path: Path,
    ) -> AppState:
        if app_state.is_loaded:
            raise RuntimeError(
                "Cannot call load_state_dict twice on the same AppState object. " "State dict has already been loaded."
            )

        DCPCheckpointLoading.load_checkpoint_(app_state=app_state, checkpoint_directory_path=checkpoint_directory_path)
        return app_state
