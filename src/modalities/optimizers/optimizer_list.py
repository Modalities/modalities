# This file contains code adapted from:
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/optimizer.py
# which is licensed under the BSD 3-Clause "New" or "Revised" License:
# https://github.com/pytorch/torchtitan/blob/main/LICENSE

import functools
from typing import Any, Iterable

from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class OptimizersList(Optimizer, Stateful, list[Optimizer]):
    """Class to handle multiple optimizers for different model parts.
    Particular relevant for pipeline parallelism, where each stage has its own optimizer.
    This class wraps a list of optimizers and provides a unified interface to step, zero_grad,
    state_dict and load_state_dict.
    """

    def __init__(self, model_parts: Iterable[nn.Module], optimizers: Iterable[Optimizer]):
        list.__init__(self, optimizers)
        self._model_parts = list(model_parts)
        assert len(self) > 0, "OptimizersList requires at least one optimizer"
        assert len(self._model_parts) == len(self), "Number of model parts must match number of optimizers"
        all_params: ParamsT = [p for model in self._model_parts for p in model.parameters() if p.requires_grad]
        Optimizer.__init__(self, all_params, dict())

    def step(self, *args, **kwargs):
        for optimizer in self:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        for optimizer in self:
            optimizer.zero_grad(*args, **kwargs)

    @staticmethod
    def _has_native_state_dict(optimizer: Optimizer) -> bool:
        """Whether an optimizer manages its own (non-torch-standard) checkpoint serialization.

        torch's ``get_optimizer_state_dict`` assumes the standard ``{"state", "param_groups"}``
        layout. Optimizers like ``EPAdamW`` (expert parallelism) wrap several sub-optimizers with a
        custom ``state_dict`` and would raise ``KeyError: 'state'`` if routed through that helper, so
        we detect them (duck-typed on their EP marker) and use their native ``state_dict``/
        ``load_state_dict`` instead -- the same path used when EP runs without pipeline parallelism.
        """
        return hasattr(optimizer, "_all_ep_params")

    def state_dict(self) -> dict[str, Any]:
        if any(self._has_native_state_dict(optimizer) for optimizer in self):
            # Namespace each stage's optimizer state by its position so keys never collide across
            # multiple stages held by the same rank (interleaved pipeline schedules).
            return {str(idx): optimizer.state_dict() for idx, optimizer in enumerate(self)}

        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self._model_parts, self) for k, v in sd.items()}

    def load_state_dict(self, state_dict: dict[str, Any]):
        if any(self._has_native_state_dict(optimizer) for optimizer in self):
            for idx, optimizer in enumerate(self):
                optimizer.load_state_dict(state_dict[str(idx)])
            return

        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self._model_parts, self))
