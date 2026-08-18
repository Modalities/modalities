import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer

from modalities.optimizers.optimizer_factory import _build_optimizer_groups_via_weight_decay_split


def _get_ep_param_ids(model: Module) -> set:
    return {id(p) for m in model.modules() if getattr(m, "_ep_enabled", False) for p in m.parameters(recurse=False)}


def _get_dense_optimizer_groups(model, ep_param_ids, weight_decay, weight_decay_groups_excluded):
    weight_decay_groups = model.weight_decay_groups
    params = {
        name: p
        for name, p in model.named_parameters()
        if p.requires_grad and id(p) not in ep_param_ids and (not isinstance(p, DTensor) or p.to_local().numel() > 0)
    }
    return _build_optimizer_groups_via_weight_decay_split(
        weight_decay, weight_decay_groups_excluded, weight_decay_groups, params
    )


class EPAdamW(Optimizer):
    """
    ZeRO stage-1 for EP (DTensor) params + standard AdamW for dense params.

    Each dp_shard rank stores optimizer states for 1/dp_shard of the EP params.
    After each step, updated EP param values are broadcast from owner to all ranks.
    Dense params are handled by a separate AdamW (FSDP2 shards them independently).
    """

    def __init__(
        self,
        model: Module,
        device_mesh,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        weight_decay_groups_excluded: list[str],
    ):
        self._dp_mesh = device_mesh["dp_shard"]
        self._dp_group = self._dp_mesh.get_group()
        self._dp_rank = dist.get_rank(self._dp_group)
        self._dp_size = dist.get_world_size(self._dp_group)

        ep_param_ids = _get_ep_param_ids(model)
        self._all_ep_params = [p for p in model.parameters() if id(p) in ep_param_ids]

        # rank r owns params[r::dp_size]
        self._owned_ep_params = self._all_ep_params[self._dp_rank :: self._dp_size]

        dense_groups = _get_dense_optimizer_groups(model, ep_param_ids, weight_decay, weight_decay_groups_excluded)

        if self._owned_ep_params:
            self._ep_adamw = AdamW(self._owned_ep_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        else:
            self._ep_adamw = None
        self._dense_adamw = AdamW(dense_groups, lr=lr, betas=betas, eps=eps)

        # unified param groups for lr_scheduler compatibility:
        # group 0 = all EP params, groups 1+ = dense weight-decay split
        ep_group = {"params": self._all_ep_params, "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        all_groups = [ep_group] + [{**g, "lr": lr, "betas": betas, "eps": eps} for g in dense_groups]
        super().__init__(all_groups, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # all-reduce
        for p in self._all_ep_params:
            if p.grad is None:
                continue
            if isinstance(p.grad, DTensor):
                local_g = p.grad.to_local()
                dist.all_reduce(local_g, op=dist.ReduceOp.SUM, group=self._dp_group)
                local_g.div_(self._dp_size)
            else:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self._dp_group)
                p.grad.div_(self._dp_size)

        # Sync lr
        if self._ep_adamw is not None:
            self._ep_adamw.param_groups[0]["lr"] = self.param_groups[0]["lr"]
        for i, group in enumerate(self._dense_adamw.param_groups):
            group["lr"] = self.param_groups[i + 1]["lr"]

        # Update ep params
        if self._ep_adamw is not None:
            self._ep_adamw.step()

        # Update dense params
        self._dense_adamw.step()

        # broadcast updated EP param local tensors
        for i, p in enumerate(self._all_ep_params):
            owner_local_rank = i % self._dp_size
            owner_global_rank = dist.get_global_rank(self._dp_group, owner_local_rank)
            if isinstance(p, DTensor):
                local_tensor = p.to_local()
            elif isinstance(p.data, DTensor):
                local_tensor = p.data.to_local()
            else:
                local_tensor = p.data
            dist.broadcast(local_tensor, src=owner_global_rank, group=self._dp_group)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for p in self._all_ep_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        self._dense_adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        # Materialize the (still zero) optimizer state of the sub-optimizers before serializing, so
        # that the returned state dict already carries the full set of moment keys even before the
        # first step. This matters for LOADING: torch's DCP derives its read plan from the local
        # state_dict(), so a freshly built optimizer whose state dict is empty would read nothing and
        # silently resume with zeroed Adam moments. torch's `get_optimizer_state_dict` does the same
        # (via its private `_init_optim_state`), but this optimizer bypasses that helper because of
        # its custom, non-standard state-dict layout.
        if self._ep_adamw is not None:
            _init_optimizer_state_(self._ep_adamw)
        _init_optimizer_state_(self._dense_adamw)
        return {
            "ep_adamw": self._ep_adamw.state_dict() if self._ep_adamw is not None else {},
            "dense_adamw": self._dense_adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if self._ep_adamw is not None and state_dict["ep_adamw"]:
            self._ep_adamw.load_state_dict(state_dict["ep_adamw"])
        self._dense_adamw.load_state_dict(state_dict["dense_adamw"])


def _init_optimizer_state_(optimizer: Optimizer) -> None:
    """Initializes an optimizer's state in-place by taking a zero-gradient, zero-learning-rate step.

    Mirrors torch's ``torch.distributed.checkpoint.state_dict._init_optim_state``: the learning rate
    is temporarily set to zero so that optimizers which update parameters irrespective of the
    gradient (e.g. AdamW's decoupled weight decay) leave the parameters untouched. Does nothing if
    the state already exists, or if gradients are present (i.e. mid-training), in which case the
    step would not be a no-op.

    Note that this operates on plain per-rank sub-optimizers only and therefore triggers no
    collectives.
    """
    if optimizer.state:
        return
    if any(param.grad is not None for group in optimizer.param_groups for param in group["params"]):
        return

    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

    original_lrs = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = torch.tensor(0.0) if isinstance(group["lr"], torch.Tensor) else 0.0
    try:
        optimizer.step()
    finally:
        for group, lr in zip(optimizer.param_groups, original_lrs):
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)


def get_ep_adam_w(
    wrapped_model,
    device_mesh,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    weight_decay_groups_excluded: list[str],
) -> EPAdamW:
    return EPAdamW(
        model=wrapped_model,
        device_mesh=device_mesh,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        weight_decay_groups_excluded=weight_decay_groups_excluded,
    )
