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

from modalities.utils.logging import get_logger


class StatefulComponents(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


class AppState(Stateful):
    """
    This is a useful wrapper for checkpointing the application state (i.e., model, optimizer, lr scheduler).
    Since this object is compliant with the Stateful protocol, DCP will automatically call
    state_dict/load_stat_dict as needed in the dcp.save/load APIs.

    Note: We take advantage of this wrapper to call distributed state dict methods on the model
    and optimizer.
    Note: this class has been copied and adapted from
    https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        """Initializes the AppState object.

        Args:
            model (nn.Module, optional): The model can be either a non-sharded model, FSDP1 or FSDP2 model.
            optimizer (Optimizer, optional): The optimizer can be either a non-sharded optimizer,
                FSDP1 or FSDP2 optimizer.
            lr_scheduler (LRScheduler, optional): The lr scheduler used during training. Defaults to None.
        """
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Returns whether the state dict has been loaded.
        Returns:
            bool: Flag indicating whether the state dict has been loaded.
        """
        return self._is_loaded

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, model: nn.Module) -> None:
        """Sets the model in the AppState object.

        Args:
            model (nn.Module): The model to set in the AppState object.
        """
        self._model = model

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Sets the optimizer in the AppState object.

        Args:
            optimizer (Optimizer): The optimizer to set in the AppState object.
        """
        self._optimizer = optimizer

    @property
    def lr_scheduler(self) -> LRScheduler:
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, lr_scheduler: LRScheduler) -> None:
        """Sets the learning rate scheduler in the AppState object.

        Args:
            lr_scheduler (LRScheduler): The learning rate scheduler to set in the AppState object.
        """
        self._lr_scheduler = lr_scheduler

    def state_dict(self) -> dict[str, Any]:
        """Returns the state dict of the AppState object.

        Returns:
            dict[str, Any]: The state dict of the AppState object.
        """
        # this line automatically manages FSDP FQN's, as well as sets the default
        # state dict type to FSDP.SHARDED_STATE_DICT
        # model_state_dict, optimizer_state_dict = get_state_dict(self._model, self._optimizer)
        sd = {}
        if self._model is not None:
            sd[StatefulComponents.MODEL.value] = ModelStateRetriever.get_state_dict(app_state=self)

        if self._optimizer is not None:
            sd[StatefulComponents.OPTIMIZER.value] = OptimizerStateRetriever.get_state_dict(app_state=self)

        if self._lr_scheduler is not None:
            sd[StatefulComponents.LR_SCHEDULER.value] = LRSchedulerStateRetriever.get_state_dict(app_state=self)
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state dict into the AppState object.

        Args:
            state_dict (dict[str, Any]): The state dict to load into the AppState object.

        Raises:
            RuntimeError: If the state dict has already been loaded.
        """
        # sets our state dicts on the model, optimizer and lr scheduler.
        if self._is_loaded:
            raise RuntimeError(
                "Cannot call load_state_dict twice on the same AppState object. " "State dict has already been loaded."
            )

        if self._model is not None:
            ModelStateRetriever.load_state_dict_(app_state=self, state_dict=state_dict[StatefulComponents.MODEL.value])

        if self._optimizer is not None:
            if StatefulComponents.OPTIMIZER.value in state_dict:
                OptimizerStateRetriever.load_state_dict_(
                    app_state=self,
                    state_dict=state_dict[StatefulComponents.OPTIMIZER.value],
                )
            else:
                get_logger(name="app_state").warning(
                    "Did not load optimizer checkpoint! "
                    f"Optimizer state dict not found in state_dict: {state_dict.keys()}."
                )

        if self._lr_scheduler is not None:
            if StatefulComponents.LR_SCHEDULER.value in state_dict:
                LRSchedulerStateRetriever.load_state_dict_(
                    app_state=self, state_dict=state_dict[StatefulComponents.LR_SCHEDULER.value]
                )
            else:
                get_logger(name="app_state").warning(
                    "Did not load lr scheduler checkpoint! "
                    f"LR scheduler state dict not found in state_dict: {state_dict.keys()}."
                )

        self._is_loaded = True


class StateRetrieverIF(ABC):
    """State retriever interface for loading and getting state dicts of
    models, optimizers and lr schedulers. Other stateful components can be added as needed
    by having the retriever implement this interface.
    """

    @staticmethod
    @abstractmethod
    def load_state_dict_(app_state: AppState, state_dict: dict[str, Any]) -> None:
        """Loads the state dict into the AppState object.

        Args:
            app_state (AppState): The application state object.
            state_dict (dict[str, Any]): The state dict to load into the AppState object.

        Raises:
            NotImplementedError: This abstract method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_state_dict(app_state: AppState) -> dict[str, Any]:
        """Returns the state dict of the AppState object.

        Args:
            app_state (AppState): The application state object.

        Raises:
            NotImplementedError: This abstract method is not implemented and should be overridden in a subclass.

        Returns:
            dict[str, Any]: The state dict of the AppState object.
        """
        raise NotImplementedError


class ModelStateRetriever(StateRetrieverIF):
    @staticmethod
    def get_state_dict(app_state: AppState) -> dict[str, Any]:
        """Returns the state dict of the model in the AppState object.

        Args:
            app_state (AppState): The app_state object containing the model.

        Returns:
            dict[str, Any]: The state dict of the model in the AppState object.
        """
        return get_model_state_dict(model=app_state.model)

    @staticmethod
    def load_state_dict_(app_state: AppState, state_dict: dict[str, Any]) -> None:
        """Loads the state dict into the model in the AppState object.

        Args:
            app_state (AppState): The app_state object containing the model.
            state_dict (dict[str, Any]): The state dict to load into the model.
        """
        set_model_state_dict(model=app_state.model, model_state_dict=state_dict, options=StateDictOptions(strict=False))


class OptimizerStateRetriever(StateRetrieverIF):
    @staticmethod
    def get_state_dict(app_state: AppState) -> dict[str, Any]:
        """Returns the state dict of the optimizer in the AppState object.

        Args:
            app_state (AppState): The app_state object containing the optimizer.

        Returns:
            dict[str, Any]: The state dict of the optimizer in the AppState object.
        """
        sd = get_optimizer_state_dict(
            model=app_state.model,
            optimizers=app_state.optimizer,
            # NOTE: Flattening is required for pipeline parallelism to work correctly.
            # see https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/components/checkpoint.py#L193-L214
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return sd

    @staticmethod
    def load_state_dict_(app_state: AppState, state_dict: dict[str, Any]) -> None:
        """Loads the state dict into the optimizer in the AppState object.

        Args:
            app_state (AppState): The app_state object containing the optimizer.
            state_dict (dict[str, Any]): The state dict to load into the optimizer.
        """
        set_optimizer_state_dict(
            model=app_state.model,
            optimizers=app_state.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


class LRSchedulerStateRetriever(StateRetrieverIF):
    @staticmethod
    def get_state_dict(app_state: AppState) -> dict[str, Any]:
        """Returns the state dict of the lr scheduler in the AppState object.

        Args:
            app_state (AppState): The app_state object containing the lr scheduler.

        Returns:
            dict[str, Any]: The state dict of the lr scheduler in the AppState object.
        """
        return app_state.lr_scheduler.state_dict()

    @staticmethod
    def load_state_dict_(app_state: AppState, state_dict: dict[str, Any]) -> None:
        """Loads the state dict into the lr scheduler in the AppState object.

        Args:
            app_state (AppState): The app_state object containing the lr scheduler.
            state_dict (dict[str, Any]): The state dict to load into the lr scheduler.
        """
        # NOTE from torchtitan:
        # https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/components/optimizer.py#L363
        # The key value we're concerned
        # within ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer
        # that is immutable. As long as ``steps`` and ``warmup_steps``
        # in ``job_config`` remain unchanged when resuming from a checkpoint, this
        # approach is safe. We call ``copy()`` here to ensure extra safety.
        app_state.lr_scheduler.load_state_dict(copy.deepcopy(state_dict))
