from enum import Enum
from typing import Any

import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.checkpointing.stateful.state_retriever import (
    LRSchedulerStateRetriever,
    ModelStateRetriever,
    OptimizerStateRetriever,
)


class StatefulComponents(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


class AppState(Stateful):
    """
    Note: this class has been copied from https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

    This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler):
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._is_loaded = False

    @property
    def is_loaded(self):
        return self._is_loaded

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> LRScheduler:
        return self._lr_scheduler

    def state_dict(self) -> dict[str, Any]:
        # this line automatically manages FSDP FQN's, as well as sets the default
        # state dict type to FSDP.SHARDED_STATE_DICT
        # model_state_dict, optimizer_state_dict = get_state_dict(self._model, self._optimizer)
        return {
            StatefulComponents.MODEL.value: ModelStateRetriever.get_state_dict(model=self._model),
            StatefulComponents.OPTIMIZER.value: OptimizerStateRetriever.get_state_dict(
                model=self._model, optimizer=self._optimizer
            ),
            StatefulComponents.LR_SCHEDULER.value: LRSchedulerStateRetriever.get_state_dict(
                lr_scheduler=self._lr_scheduler
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # sets our state dicts on the model, optimizer and lr scheduler.
        if self._is_loaded:
            raise RuntimeError(
                "Cannot call load_state_dict twice on the same AppState object. " "State dict has already been loaded."
            )

        ModelStateRetriever.load_state_dict_(model=self._model, state_dict=state_dict[StatefulComponents.MODEL.value])
        OptimizerStateRetriever.load_state_dict_(
            model=self._model,
            optimizer=self._optimizer,
            state_dict=state_dict[StatefulComponents.OPTIMIZER.value],
        )
        LRSchedulerStateRetriever.load_state_dict_(
            lr_scheduler=self._lr_scheduler, state_dict=state_dict[StatefulComponents.LR_SCHEDULER.value]
        )
        self._is_loaded = True
