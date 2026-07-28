"""Auxiliary-loss-free load balancing for mixture-of-experts layers.

Implements the update rule of "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"
(https://arxiv.org/abs/2408.15664), which Nemotron-3 Nano uses as its primary balancing mechanism.

The idea: every expert carries an additive bias that is applied only when *selecting* experts, not
when weighting their outputs. After each optimizer step the bias of an under-loaded expert is nudged
up and that of an over-loaded expert down, by a fixed step size. Because the update depends only on
the *sign* of the load deviation, it is robust to the exact token counts and adds no gradient term
to the loss - unlike the classic auxiliary loss, it does not fight the language modelling objective.

Two details matter for correctness in a distributed training loop:

* The update must happen once per **optimizer step**, not per micro-batch, and it must see the load
  summed over all data-parallel ranks. A rank-local update would let ranks drift apart, and a
  per-micro-batch update would over-correct by the gradient accumulation factor.
* The token counters are reset after each update so that the next step measures a fresh window.
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from modalities.models.components.moe.moe import MoE
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees

logger = logging.getLogger(__name__)


def get_moe_layers(model: nn.Module) -> list[MoE]:
    """
    Collects every mixture-of-experts block inside a model.

    Args:
        model (nn.Module): The model to search. May be wrapped (FSDP1/FSDP2) or a pipeline stage.

    Returns:
        list[MoE]: The MoE blocks found, in module traversal order.
    """
    return [module for module in model.modules() if isinstance(module, MoE)]


# Name under which the non-pipeline mesh dimensions are flattened into a single reduction group.
_REDUCTION_MESH_DIM_NAME = "moe_load_balancing"


def get_expert_load_reduction_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    """
    Resolves the process group over which expert token counts have to be summed.

    All mesh dimensions except pipeline parallelism are included:

    * Data-parallel and context-parallel ranks each see different tokens, so their counts must be
      added to obtain the true expert load.
    * Tensor-parallel ranks see the *same* tokens. Including them scales every expert's count by the
      same factor, and since the update rule only looks at the sign of the deviation from the mean,
      a uniform factor is harmless. Including them keeps the collective simple.
    * Pipeline-parallel ranks hold *different layers*. A given MoE layer exists on exactly one
      stage, so a collective across pipeline ranks would try to match tensors that do not exist on
      the other stages. Pipeline ranks must therefore be excluded.

    Args:
        device_mesh (DeviceMesh | None): The device mesh, or None when running without one.

    Returns:
        dist.ProcessGroup | None: The group to reduce over, or None if no reduction is needed.
    """
    if not dist.is_available() or not dist.is_initialized():
        return None
    if device_mesh is None:
        return dist.group.WORLD

    mesh_dim_names = tuple(device_mesh.mesh_dim_names or ())
    reduction_dims = tuple(name for name in mesh_dim_names if name != ParallelismDegrees.PP.value)
    if not reduction_dims:
        # Pipeline parallelism only: every rank owns its layers alone, nothing to reduce.
        return None
    if len(reduction_dims) == len(mesh_dim_names):
        return dist.group.WORLD

    submesh = device_mesh[reduction_dims]
    if len(reduction_dims) == 1:
        return submesh.get_group()
    try:
        return submesh._flatten(_REDUCTION_MESH_DIM_NAME).get_group()
    except RuntimeError:
        # Already flattened under this name by an earlier call.
        return device_mesh[_REDUCTION_MESH_DIM_NAME].get_group()


@torch.no_grad()
def update_expert_biases(
    moe_layers: list[MoE],
    update_rate: float,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """
    Applies one auxiliary-loss-free load balancing step to every MoE layer.

    For each layer the per-expert token counts are summed across the data-parallel group, compared
    against the mean load, and the expert bias is moved by ``update_rate`` in the direction that
    equalizes the load. Counters are reset afterwards.

    Args:
        moe_layers (list[MoE]): The MoE blocks to update.
        update_rate (float): The step size of the bias update. Nemotron-3 Nano uses 1e-3.
        process_group (dist.ProcessGroup | None): The group across which token counts are summed.
            None means no reduction (single process, or already-global counts).
    """
    for moe in moe_layers:
        router = moe.router
        if router.expert_bias is None:
            continue

        tokens_per_expert = router.tokens_per_expert
        if process_group is not None:
            dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM, group=process_group)

        if tokens_per_expert.sum() == 0:
            # No tokens were routed since the last update (e.g. an evaluation-only interval).
            continue

        # Positive where an expert is under-loaded, negative where it is over-loaded.
        load_deviation = tokens_per_expert.mean() - tokens_per_expert
        router.expert_bias += torch.sign(load_deviation) * update_rate
        router.tokens_per_expert.zero_()


class MoEBalancing:
    """Factory for the load-balancing optimizer hook."""

    @staticmethod
    def register_expert_bias_update_hook(
        optimizer: Optimizer,
        model: nn.Module,
        expert_bias_update_rate: float,
        device_mesh: DeviceMesh | None = None,
    ) -> Optimizer:
        """
        Attaches the expert bias update to an optimizer and returns the same optimizer.

        This is an optimizer *decorator*: it is registered as an ``optimizer`` component variant so
        that a config can wrap a plain Adam/AdamW without any change to the training loop. Using an
        optimizer step pre-hook is what pins the update to one per optimizer step, making it correct
        under gradient accumulation.

        Args:
            optimizer (Optimizer): The optimizer to decorate.
            model (nn.Module): The model whose MoE layers should be balanced.
            expert_bias_update_rate (float): The step size of the bias update.
            device_mesh (DeviceMesh | None): Device mesh used to resolve the data-parallel group.

        Raises:
            ValueError: If the update rate is not positive.

        Returns:
            Optimizer: The same optimizer instance, with the hook registered.
        """
        if expert_bias_update_rate <= 0:
            raise ValueError(f"expert_bias_update_rate must be positive, got {expert_bias_update_rate}.")

        moe_layers = get_moe_layers(model)
        if not moe_layers:
            logger.warning(
                "No MoE layers found in the model; the expert bias update hook will not do anything. "
                "(This is expected for a pipeline stage that holds no MoE layers.)"
            )
            return optimizer

        balanced_layers = [moe for moe in moe_layers if moe.router.expert_bias is not None]
        if not balanced_layers:
            logger.warning(
                "Found %d MoE layers but none of them maintains an expert bias "
                "(use_expert_bias=False); auxiliary-loss-free load balancing is disabled.",
                len(moe_layers),
            )
            return optimizer

        process_group = get_expert_load_reduction_group(device_mesh)

        def _hook(opt: Optimizer, args: tuple, kwargs: dict) -> None:
            del opt, args, kwargs
            update_expert_biases(
                moe_layers=balanced_layers,
                update_rate=expert_bias_update_rate,
                process_group=process_group,
            )

        optimizer.register_step_pre_hook(_hook)
        logger.info(
            "Registered auxiliary-loss-free load balancing for %d MoE layers (update rate %g).",
            len(balanced_layers),
            expert_bias_update_rate,
        )
        return optimizer
