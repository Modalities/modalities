from typing import Dict, List, Tuple

import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Adam, AdamW, Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.models.components.layer_norms import LayerNorms
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
        # preparation
        groups = model.module.optimizer_module_groups  # e.g. for GPT2: ["linear", "embedding", "layernorm"]
        for group in weight_decay_excluded:  # e.g. for GPT2: ["embedding", "layernorm"]
            assert group in groups, f"group = {group} specified in weight_decay_excluded is not defined."

        modules, params, num_modules, num_params = {}, {}, {}, {}
        weight_decay_group = {group: weight_decay if group not in weight_decay_excluded else 0.0 for group in groups}

        # create module groups
        modules["all"] = {name: module for name, module in model.named_modules()}
        modules["linear"] = {
            name: module
            for name, module in modules["all"].items()
            if type(module) == nn.Linear and not name.endswith("lm_head")
        }
        modules["embedding"] = {name: module for name, module in modules["all"].items() if type(module) == nn.Embedding}
        modules["layernorm"] = {
            name: module for name, module in modules["all"].items() if type(module) in [e.value for e in LayerNorms]
        }

        if 0:  # This is for debugging and to help define new models
            print_module_names_for_groups(modules, groups)

        # create parameter groups
        param_dict = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}

        # TODO: make this less error-prone and don't rely on existence of bias keys
        params["linear"] = [
            [param_dict[f"{name}.weight"], param_dict[f"{name}.bias"]] for name in modules["linear"].keys()
        ]
        params["embedding"] = [[param_dict[f"{name}.weight"]] for name in modules["embedding"].keys()]
        params["layernorm"] = [
            [param_dict[f"{name}.weight"], param_dict[f"{name}.bias"]] for name in modules["layernorm"].keys()
        ]

        for group in groups:
            params[group], num_params[group] = flatten_and_count(params[group])

        # create optimizer parameter groups
        optim_groups = [{"params": params[group], "weight_decay": weight_decay_group[group]} for group in groups]

        # print overview & check total number of parameters
        num_modules["all"] = sum([len(modules[group]) for group in groups])
        num_params["all"] = sum([num_params[group] for group in groups])
        print(
            f"{num_modules['all']} modules with {num_params['all']:,} parameters"
            + "were split into the following optimizer groups:"
        )
        for group in groups:
            print(
                f"{group} ({len(modules[group])} modules with {num_params[group]:,} parameters): "
                + "weight_decay = {weight_decay_group[group]}"
            )

        num_params_check = compute_number_of_trainable_parameters(model)
        assert num_params["all"] == num_params_check, (
            f"ERROR! Inconsistent number of parameters (found {num_params['all']}, "
            + "should be {num_params_check}) after split into optimizer parameter groups."
        )

    return optim_groups


def print_module_names_for_groups(modules, groups):
    for group in groups:
        print(f"module names of group={group}:")
        for name in modules[group].keys():
            print(name)


def flatten_and_count(params):
    params = [elem for sublist in params for elem in sublist]
    num_params = sum(p.numel() for p in params)
    return params, num_params
