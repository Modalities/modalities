import re
from pathlib import Path

import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Adam, AdamW, Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.exceptions import OptimizerError
from modalities.models.model import NNModel
from modalities.util import get_local_number_of_trainable_parameters, print_rank_0

OptimizerGroups = list[dict[str, list[nn.Parameter] | float]]


class OptimizerFactory:
    def get_adam(
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        weight_decay_groups_excluded: list[str],
        wrapped_model: nn.Module,
    ) -> Optimizer:
        optimizer_groups = get_optimizer_groups(wrapped_model, weight_decay, weight_decay_groups_excluded)
        optimizer = Adam(params=optimizer_groups, lr=lr, betas=betas, eps=eps)
        return optimizer

    def get_adam_w(
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        weight_decay_groups_excluded: list[str],
        wrapped_model: nn.Module,
    ) -> Optimizer:
        optimizer_groups = get_optimizer_groups(wrapped_model, weight_decay, weight_decay_groups_excluded)
        optimizer = AdamW(params=optimizer_groups, lr=lr, betas=betas, eps=eps)
        return optimizer

    @staticmethod
    def get_checkpointed_optimizer(
        checkpoint_loading: CheckpointLoadingIF, checkpoint_path: Path, wrapped_model: nn.Module, optimizer: Optimizer
    ) -> Optimizer:
        wrapped_optimizer = checkpoint_loading.load_optimizer_checkpoint(
            file_path=checkpoint_path, optimizer=optimizer, model=wrapped_model
        )
        return wrapped_optimizer


def get_optimizer_groups(model: FSDP, weight_decay: float, weight_decay_groups_excluded: list[str]) -> OptimizerGroups:
    """
    divide model parameters into optimizer groups (with or without weight decay)

    inspired by:
    - https://github.com/pytorch/pytorch/issues/101343
    - https://github.com/karpathy/nanoGPT
    """
    if weight_decay == 0 or len(weight_decay_groups_excluded) == 0:
        # there will be 1 optimizer group, i.e. all parameters have the same weight decay
        optimizer_groups = [{"params": list(model.parameters()), "weight_decay": weight_decay}]
        optimizer_groups_names = ["all"]
    else:
        # there will be N optimizer groups, i.e. one for each model parameter group
        _assert_existence_of_weight_decay_groups_excluded(model, weight_decay_groups_excluded)
        optimizer_groups, optimizer_groups_names = _create_optimizer_groups(
            model, weight_decay, weight_decay_groups_excluded
        )

    _assert_completeness_of_optimizer_groups(model, optimizer_groups)
    _print_optimizer_groups_overview(optimizer_groups, optimizer_groups_names)
    return optimizer_groups


def _assert_existence_of_weight_decay_groups_excluded(model: FSDP, weight_decay_groups_excluded: list[str]) -> None:
    """
    checks the existence of all groups
    that are to be excluded from weight decay

    Example GPT2:
        weight_decay_groups = {"linear": [".attn", ".mlp"], "embedding": [".wte", ".wpe"], "layernorm": [".*_norm"]]
        weight_decay_groups_excluded = ["embedding", "layernorm"]
    """
    nn_model: NNModel = model.module
    weight_decay_groups = nn_model.weight_decay_groups
    for group in weight_decay_groups_excluded:
        if group not in weight_decay_groups.keys():
            raise OptimizerError(
                f"group = {group} specified in weight_decay_groups_excluded is not "
                + f"in models optimizer_module_groups = {list(weight_decay_groups.keys())}"
            )


def _create_optimizer_groups(
    model: FSDP, weight_decay: float, weight_decay_groups_excluded: list[str]
) -> tuple[OptimizerGroups, list[str]]:
    """
    create optimizer groups of parameters with different weight decays that are to be used in Adam or AdamW
    """
    nn_model: NNModel = model.module
    weight_decay_groups = nn_model.weight_decay_groups
    params = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}

    if (
        False
    ):  # This is for debugging only, and may serve as a convenient helper tool during the development of new models
        _print_params(params)

    optimizer_groups = [
        {
            "params": _filter_params_for_weight_decay_group(params, regex_expressions=weight_decay_groups[group]),
            "weight_decay": weight_decay if group not in weight_decay_groups_excluded else 0.0,
        }
        for group in weight_decay_groups.keys()
    ]
    return optimizer_groups, weight_decay_groups.keys()


def _filter_params_for_weight_decay_group(
    params: dict[str, list[nn.Parameter]], regex_expressions: list[str]
) -> list[nn.Parameter]:
    """
    filter parameters by their name.
    a parameter is kept if and only if it contains at least one of the regex expressions.
    """
    return [
        parameter
        for name, parameter in params.items()
        if any([bool(re.search(regex_expression, name)) for regex_expression in regex_expressions])
    ]


def _print_params(params) -> None:
    """
    for debugging only
    """
    for i, name in enumerate(params.keys()):
        print_rank_0(f"{i + 1} {name}")


def _print_optimizer_groups_overview(optimizer_groups: OptimizerGroups, optimizer_groups_names: list[str]) -> None:
    """
    for each optimizer group, the following is printed:
        - the number of modules
        - the number of parameters
        - the weight decay
    """
    assert len(optimizer_groups) == len(optimizer_groups_names)
    num_modules_all, num_params_all = 0, 0
    print_rank_0("=> optimizer groups:")
    for optimizer_group, optimizer_group_name in zip(optimizer_groups, optimizer_groups_names):
        num_modules = len(optimizer_group["params"])
        num_params = sum(parameter.numel() for parameter in optimizer_group["params"])
        print_rank_0(
            f"{optimizer_group_name} ({num_modules} modules with {num_params:,} parameters): "
            f"weight_decay = {optimizer_group['weight_decay']}"
        )
        num_modules_all += num_modules
        num_params_all += num_params
    print_rank_0(f"=> all ({num_modules_all} modules with {num_params_all:,} parameters)")


def _assert_completeness_of_optimizer_groups(model: FSDP, optimizer_groups: OptimizerGroups) -> None:
    """
    checks that the number of parameters in the optimizer groups
    sum up to the total number of model parameters as expected
    """
    num_params_check = get_local_number_of_trainable_parameters(model)
    num_params = sum(p.numel() for optimizer_group in optimizer_groups for p in optimizer_group["params"])
    if num_params != num_params_check:
        raise OptimizerError(
            f"ERROR! Inconsistent number of parameters (found {num_params}, "
            + f"should be {num_params_check}) after split into optimizer parameter groups."
        )
