# This file contains code adapted from:
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/lr_scheduler.py
# which is licensed under the BSD 3-Clause "New" or "Revised" License:
# https://github.com/pytorch/torchtitan/blob/main/LICENSE

import copy
from typing import Any, Iterable

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler


class SchedulerList(LRScheduler, Stateful, list[LRScheduler]):
    """A list of learning rate schedulers that can be treated as a single scheduler.
    Each scheduler in the list should correspond to an optimizer in a multi-optimizer setup.
    NOTE: Similar to torchtitan, this class assumes that all schedulers have the same state.
    """

    def __init__(self, schedulers: Iterable[LRScheduler]):
        list.__init__(self, schedulers)
        assert len(self) > 0, "SchedulerList requires at least one scheduler"

    def state_dict(self) -> dict[str, Any]:
        return self[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for scheduler in self:
            scheduler.load_state_dict(copy.deepcopy(state_dict))

    def get_last_lr(self):
        return self[0].get_last_lr()

    def get_lr(self):
        return self[0].get_lr()

    def step(self, epoch: int | None = None):
        for scheduler in self:
            scheduler.step(epoch)

    @property
    def base_lrs(self):
        return self[0].base_lrs

    @property
    def last_epoch(self):
        return self[0].last_epoch
