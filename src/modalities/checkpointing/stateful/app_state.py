import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


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

    def __init__(self, model: nn.Module, optimizer: Optimizer, lr_scheduler: Optional[LRScheduler] = None):
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
        sd = {
            StatefulComponents.MODEL.value: ModelStateRetriever.get_state_dict(app_state=self),
            StatefulComponents.OPTIMIZER.value: OptimizerStateRetriever.get_state_dict(
                app_state=self,
            ),
        }
        if self._lr_scheduler is not None:
            sd[StatefulComponents.LR_SCHEDULER.value] = LRSchedulerStateRetriever.get_state_dict(app_state=self)
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # sets our state dicts on the model, optimizer and lr scheduler.
        if self._is_loaded:
            raise RuntimeError(
                "Cannot call load_state_dict twice on the same AppState object. " "State dict has already been loaded."
            )

        ModelStateRetriever.load_state_dict_(app_state=self, state_dict=state_dict[StatefulComponents.MODEL.value])
        OptimizerStateRetriever.load_state_dict_(
            app_state=self,
            state_dict=state_dict[StatefulComponents.OPTIMIZER.value],
        )
        if self._lr_scheduler is not None:
            LRSchedulerStateRetriever.load_state_dict_(
                app_state=self, state_dict=state_dict[StatefulComponents.LR_SCHEDULER.value]
            )
        self._is_loaded = True


class StateRetrieverIF(ABC):
    @staticmethod
    @abstractmethod
    def load_state_dict_(app_state: AppState, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_state_dict(app_state: AppState) -> dict[str, Any]:
        raise NotImplementedError


class ModelStateRetriever(StateRetrieverIF):
    @staticmethod
    def get_state_dict(app_state: AppState) -> dict[str, Any]:
        return get_model_state_dict(model=app_state.model)

    @staticmethod
    def load_state_dict_(app_state: AppState, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(model=app_state.model, model_state_dict=state_dict, options=StateDictOptions(strict=False))


class OptimizerStateRetriever(StateRetrieverIF):
    @staticmethod
    def get_state_dict(app_state: AppState) -> dict[str, Any]:
        sd = get_optimizer_state_dict(
            model=app_state.model,
            optimizers=app_state.optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return sd

    @staticmethod
    def load_state_dict_(app_state: AppState, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            model=app_state.model,
            optimizers=app_state.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


class LRSchedulerStateRetriever(StateRetrieverIF):
    @staticmethod
    def get_state_dict(app_state: AppState) -> dict[str, Any]:
        return app_state.lr_scheduler.state_dict()

    @staticmethod
    def load_state_dict_(app_state: AppState, state_dict: dict[str, Any]) -> None:
        # NOTE from torchtitan:
        # https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/components/optimizer.py#L363
        # The key value we're concerned
        # within ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer
        # that is immutable. As long as ``steps`` and ``warmup_steps``
        # in ``job_config`` remain unchanged when resuming from a checkpoint, this
        # approach is safe. We call ``copy()`` here to ensure extra safety.
        app_state.lr_scheduler.load_state_dict(copy.deepcopy(state_dict))
