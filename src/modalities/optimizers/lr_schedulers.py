import warnings

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR


class DummyLRScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        if not self._get_lr_called_within_step:  # type error expected due to internal pytorch implementation
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        return self.base_lrs


class LRSchedulerFactory:
    @staticmethod
    def get_linear_warmup_cosine_annealing_lr_scheduler(
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        initial_lr: float,
        final_lr: float,
        max_lr: float,
        last_epoch: int = -1,
    ) -> SequentialLR:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be greater than 0.")
        if total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps.")

        if not all(base_lr == max_lr for base_lr in [group["lr"] for group in optimizer.param_groups]):
            raise ValueError(
                "All parameter groups must have the same initial_lr."
                "and it must be equal to the initial_lr passed to the LR scheduler factory."
            )

        warmup_scheduler = LinearLR(
            optimizer=optimizer,
            start_factor=initial_lr / max_lr,
            end_factor=1,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=final_lr,
        )

        return SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
            last_epoch=last_epoch,
        )
