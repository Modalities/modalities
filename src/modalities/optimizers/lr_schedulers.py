import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class DummyLRScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:  # type error expected due to internal pytorch implementation
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float]:
        return self.base_lrs
