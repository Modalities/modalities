import re
from typing import Dict, List, Tuple

import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Adam, AdamW, Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.util import compute_number_of_trainable_parameters


class OptimizerFactory:
    def get_adam(
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        weight_decay_excluded: List[str],
        wrapped_model: nn.Module,
    ) -> Optimizer:
        model_parameter_groups = get_parameter_groups(wrapped_model, weight_decay, weight_decay_excluded)
        optimizer = Adam(params=model_parameter_groups, lr=lr, betas=betas, eps=eps)
        return optimizer

    def get_adam_w(
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        weight_decay_excluded: List[str],
        wrapped_model: nn.Module,
    ) -> Optimizer:
        model_parameter_groups = get_parameter_groups(wrapped_model, weight_decay, weight_decay_excluded)
        optimizer = AdamW(params=model_parameter_groups, lr=lr, betas=betas, eps=eps)
        return optimizer

    @staticmethod
    def get_checkpointed_optimizer(
        checkpoint_loading: CheckpointLoadingIF, checkpoint_path, wrapped_model: nn.Module, optimizer: Optimizer
    ) -> Optimizer:
        wrapped_optimizer = checkpoint_loading.load_optimizer_checkpoint(
            file_path=checkpoint_path, optimizer=optimizer, model=wrapped_model
        )
        return wrapped_optimizer


def get_parameter_groups(
    model: FSDP, weight_decay: float, weight_decay_excluded: List[str]
) -> List[Dict[str, List[nn.Parameter] | float]]:
    """
    divide model parameters into 2 groups, one with and one without weight decay

    inspired by:
    - https://github.com/pytorch/pytorch/issues/101343
    - https://github.com/karpathy/nanoGPT
    """

    print("[Optimizer Groups]")
    if weight_decay == 0.0 or len(weight_decay_excluded) == 0:
        # all parameters have the same weight decay and there is only 1 group
        optim_groups = [{"params": model.parameters(), "weight_decay": weight_decay}]
        print(f"all parameters have weight_decay = {optim_groups[0]['weight_decay']}")
    else:
        # example GPT2:
        # groups = {"linear": [".attn", ".mlp"], "embedding": [".wte", ".wpe"], "layernorm": [".*_norm"]]
        # weight_decay_excluded = ["embedding", "layernorm"]
        group_mapping = model.module.optimizer_module_groups
        for group in weight_decay_excluded:
            assert group in group_mapping.keys(), f"group = {group} specified in weight_decay_excluded is not defined."

        params, num_params, num_modules = {}, {}, {}
        weight_decay_group = {
            group: weight_decay if group not in weight_decay_excluded else 0.0 for group in group_mapping.keys()
        }

        # create parameter groups
        params["all"] = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        for group in group_mapping.keys():
            params[group] = {
                name: parameter
                for name, parameter in params["all"].items()
                if any([bool(re.search(regex_expression, name)) for regex_expression in group_mapping[group]])
            }

        # count parameters & modules
        for group in group_mapping.keys():
            num_modules[group] = len(params[group])
            num_params[group] = sum(p.numel() for p in params[group].values())

        # print overview
        if 0:  # This is for debugging and to help define new models
            print_parameter_names_for_groups(params, group_mapping.keys())

        num_modules["all"] = sum([num_modules[group] for group in group_mapping.keys()])
        num_params["all"] = sum([num_params[group] for group in group_mapping.keys()])
        print(
            f"{num_modules['all']} modules with {num_params['all']:,} parameters "
            + "were split into the following optimizer groups:"
        )
        for group in group_mapping.keys():
            print(
                f"{group} ({num_modules[group]} modules with {num_params[group]:,} parameters): "
                + f"weight_decay = {weight_decay_group[group]}"
            )

        # check total number of parameters
        num_params_check = compute_number_of_trainable_parameters(model)
        assert num_params["all"] == num_params_check, (
            f"ERROR! Inconsistent number of parameters (found {num_params['all']}, "
            + f"should be {num_params_check}) after split into optimizer parameter groups."
        )

        # create optimizer parameter groups
        optim_groups = [
            {"params": params[group].values(), "weight_decay": weight_decay_group[group]}
            for group in group_mapping.keys()
        ]

    return optim_groups


def print_parameter_names_for_groups(params, groups):
    for group in groups:
        print(f"parameter names of group={group}:")
        for i, name in enumerate(params[group].keys()):
            print(i + 1, name)
