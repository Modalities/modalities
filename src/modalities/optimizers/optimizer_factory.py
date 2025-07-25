import re
from pathlib import Path

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.distributed.tensor import DTensor
from torch.optim import Adam, AdamW, Optimizer

from modalities.checkpointing.checkpoint_loading import FSDP1CheckpointLoadingIF
from modalities.exceptions import OptimizerError
from modalities.models.model import NNModel
from modalities.util import get_local_number_of_trainable_parameters, print_rank_0
from modalities.utils.typing_utils import FSDPX

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
    def get_fsdp1_checkpointed_optimizer_(
        checkpoint_loading: FSDP1CheckpointLoadingIF,
        checkpoint_path: Path,
        wrapped_model: FSDP1,
        optimizer: Optimizer,
    ) -> Optimizer:
        """Loads an FSDP1-checkpointed optimizer from a checkpoint file.

        Args:
            checkpoint_loading (FSDP1CheckpointLoadingIF): The FDSP1 checkpoint loading strategy.
            checkpoint_path (Path): The path to the checkpoint file.
            wrapped_model (FSDP1): The FSDP1 model associated with the optimizer.
            optimizer (Optimizer): The optimizer to load the checkpoint into.

        Returns:
            Optimizer: The optimizer loaded from the checkpoint.
        """
        checkpoint_loading.load_optimizer_checkpoint_(
            file_path=checkpoint_path, optimizer=optimizer, model=wrapped_model
        )
        return optimizer


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


def _assert_existence_of_weight_decay_groups_excluded(
    model: nn.Module, weight_decay_groups_excluded: list[str]
) -> None:
    """
    checks the existence of all groups
    that are to be excluded from weight decay

    Example GPT2:
        weight_decay_groups = {
            "linear": [".attn", ".mlp", ".lm_head.weight"],
            "embedding": [".wte", ".wpe"], "layernorm": [".*_norm"]]
        }
        weight_decay_groups_excluded = ["embedding", "layernorm"]
    """
    # FSDP 1
    if hasattr(model, "module"):
        nn_model: NNModel = model.module
    # FSDP 2
    else:
        nn_model = model
    weight_decay_groups = nn_model.weight_decay_groups
    for group in weight_decay_groups_excluded:
        if group not in weight_decay_groups.keys():
            raise OptimizerError(
                f"group = {group} specified in weight_decay_groups_excluded is not "
                + f"in models optimizer_module_groups = {list(weight_decay_groups.keys())}"
            )


def _create_optimizer_groups(
    model: FSDPX, weight_decay: float, weight_decay_groups_excluded: list[str]
) -> tuple[OptimizerGroups, list[str]]:
    """
    create optimizer groups of parameters with different weight decays that are to be used in Adam or AdamW
    """
    # FSDP 1
    if isinstance(model, FSDP1):
        nn_model: NNModel = model.module
        weight_decay_groups = nn_model.weight_decay_groups
        params = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}

    # FSDP 2
    elif isinstance(model, FSDP2):
        nn_model = model
        weight_decay_groups = nn_model.weight_decay_groups
        params = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad and (not isinstance(param, DTensor) or param.to_local().numel() > 0)
        }

    else:
        raise OptimizerError(
            f"model {type(model)} is not an instance of FSDP1 or FSDP2. " "Please use the correct model type."
        )

    if (
        False
    ):  # This is for debugging only, and may serve as a convenient helper tool during the development of new models
        _print_params(params)

    if len(params) == 0:
        raise OptimizerError(
            f"model {type(model)} has no parameters with requires_grad=True (i.e., no traininable parameters)."
        )

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


def _assert_completeness_of_optimizer_groups(model: nn.Module, optimizer_groups: OptimizerGroups) -> None:
    """
    checks that the number of parameters in the optimizer groups
    sum up to the total number of model parameters as expected
    """
    num_params_check = get_local_number_of_trainable_parameters(model)
    if isinstance(model, FSDP1):
        num_params = sum(p.numel() for optimizer_group in optimizer_groups for p in optimizer_group["params"])
    elif isinstance(model, FSDP2):
        num_params = sum(
            (p.to_local().numel() if isinstance(p, DTensor) else p.numel())
            for optimizer_group in optimizer_groups
            for p in optimizer_group["params"]
        )
    else:
        raise OptimizerError(
            f"Model {type(model)} is not an instance of FSDP1 or FSDP2. Please use the correct model type."
        )
    if num_params != num_params_check:
        raise OptimizerError(
            f"ERROR! Inconsistent number of parameters (found {num_params}, "
            + f"should be {num_params_check}) after split into optimizer parameter groups."
        )
