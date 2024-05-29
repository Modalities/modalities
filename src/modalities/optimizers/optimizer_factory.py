from typing import Dict, List, Tuple

import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Adam, AdamW, Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF


class OptimizerFactory:
    @staticmethod
    def get_adam(
        lr: float, betas: Tuple[float, float], eps: float, weight_decay: float, wrapped_model: nn.Module
    ) -> Optimizer:
        model_parameter_groups = get_parameter_groups(wrapped_model, weight_decay)
        optimizer = Adam(params=model_parameter_groups, lr=lr, betas=betas, eps=eps)
        return optimizer

    @staticmethod
    def get_adam_w(
        lr: float, betas: Tuple[float, float], eps: float, weight_decay: float, wrapped_model: nn.Module
    ) -> Optimizer:
        model_parameter_groups = get_parameter_groups(wrapped_model, weight_decay)
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


@staticmethod
def get_parameter_groups(model: FSDP, weight_decay: float) -> List[Dict[str, List[nn.Parameter] | float]]:
    # filter out parameters that do not require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # split parameters into different groups
    nodecay_params = [parameter for name, parameter in param_dict.items() if "norm" in name]
    decay_params = [parameter for name, parameter in param_dict.items() if "norm" not in name]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # print number of parameters per group
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    return optim_groups
