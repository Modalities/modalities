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
        model_parameters = wrapped_model.parameters()
        optimizer = Adam(params=model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        return optimizer

    @staticmethod
    def get_adam_w(
        lr: float, betas: Tuple[float, float], eps: float, weight_decay: float, wrapped_model: nn.Module
    ) -> Optimizer:
        # model_parameters = wrapped_model.parameters()
        model_parameters = get_parameter_groups(wrapped_model, weight_decay)
        optimizer = AdamW(params=model_parameters, lr=lr, betas=betas, eps=eps)
        # optimizer = AdamW(params=wrapped_model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    nodecay_params = [parameter for name, parameter in param_dict.items() if "norm" in name]
    decay_params = [parameter for name, parameter in param_dict.items() if "norm" not in name]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    # use_fused = fused_available and device_type == 'cuda'
    # extra_args = {} # dict(fused=True) if use_fused else dict()
    # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    # print(f"using fused AdamW: {use_fused}")
    return optim_groups
