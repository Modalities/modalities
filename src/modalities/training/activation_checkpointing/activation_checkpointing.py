from collections import defaultdict
from functools import partial
from typing import Set

import torch
import torch.nn as nn
import torch.ops as ops
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, apply_activation_checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP1
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from modalities.config.config import ActivationCheckpointedModelConfig
from modalities.training.activation_checkpointing.activation_checkpointing_variants import (
    ActivationCheckpointingVariants,
)
from modalities.util import get_module_class_from_name, print_rank_0


def is_module_to_apply_activation_checkpointing(
    submodule: nn.Module, activation_checkpointing_modules: list[type]
) -> bool:
    return isinstance(submodule, tuple(activation_checkpointing_modules))


def apply_activation_checkpointing_fsdp1_inplace(model: FSDP1, activation_checkpointing_modules: list[str]):
    activation_checkpointing_module_types = [
        get_module_class_from_name(model, m) for m in activation_checkpointing_modules
    ]
    if not isinstance(model, FSDP1):
        raise ValueError(
            "This activation checkpointing component can only be applied to FSDP1 wrapped models. "
            "Use the respective FSDP2 component for FSDP2 models."
        )
    non_reentrant_wrapper = partial(ptd_checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT, debug=False)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: is_module_to_apply_activation_checkpointing(
            submodule, activation_checkpointing_module_types
        ),
    )


class ActivationCheckpointing:
    """In eager / normal mode, every module stores ALL activations in the forward pass and
    reuses the activations for gradient computation in the backward pass. Thus, the overall memory
    footprint of the activations accumulates during the forward pass and peaks before running the backward pass.
    During the backward pass, the activations are cleared once they are not needed anymore.

    With activation checkpointing, the regions that are ACed do only store the input and output activations, but
    no intermediate activations, thus, reducing the overall memory footprint. In the backward pass,
    the intermediate activations are recomputed, trading off a lower memory footprint with a higher compute cost.
    Typically, these ACed regions are the transformer blocks in the case of a GPT model.

    With selective AC, we add another AC variant, that allows for a more granular control over the AC process.
    This variant allows to only save the activations of certain, typically compute intensive operations, while
    recomputing the activations of all other operations. Thus, the overall memory footprint is reduced, while
    the compute cost is not increased too much.

    The implemenation is heavily inspired by the torch titan implementation:
    https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/models/llama/parallelize_llama.py#L294

    """

    SAVE_DICT = {
        # as prposed by torch titan
        # https://github.com/pytorch/torchtitan/blob/30b9ea0b1ad893379d2ff3b12dbf18600730c249/torchtitan/models/llama3/parallelize_llama.py#L218
        # This is the list of ops that are saved by default in torch titan.
        # These operations are typically compute intensive and their activations are
        # therefore saved and not recomputed in the backward pass.
        # This list differs from the compute intensive ops list in the
        # pytorch AC tutorial: https://pytorch.org/blog/activation-checkpointing-techniques/
        "ops.aten.mm.default": ops.aten.mm.default,
        "ops.aten._scaled_dot_product_efficient_attention.default": ops.aten._scaled_dot_product_efficient_attention.default,  # noqa
        "ops.aten._scaled_dot_product_flash_attention.default": ops.aten._scaled_dot_product_flash_attention.default,
        "ops._c10d_functional.reduce_scatter_tensor.default": ops._c10d_functional.reduce_scatter_tensor.default,
        # for low precision training, it's useful to always save
        # the result of max, since the absolute maximum is
        # used to compute the scaling factor for quantization.
        "ops.aten.max.default": ops.aten.max.default,
    }

    @staticmethod
    def apply_activation_checkpointing_(
        ac_variant: ActivationCheckpointingVariants,
        layers_fqn: str,
        model: nn.Module,
        ac_fun_params: (
            ActivationCheckpointedModelConfig.FullACParams
            | ActivationCheckpointedModelConfig.SelectiveLayerACParams
            | ActivationCheckpointedModelConfig.SelectiveOpACParams
        ),
    ):
        """Applies activation checkpointing to a given model. There are three variants of activation checkpointing:
        1. FULL_ACTIVATION_CHECKPOINTING: applies activation checkpointing to all layers. In thise case,
           only the inputs and outputs of each layer are saved, but not the intermediate activations.
        2. SELECTIVE_LAYER_ACTIVATION_CHECKPOINTING: applies activation checkpointing to every `ac_freq` layer.
           It is similar to FULL_ACTIVATION_CHECKPOINTING, but only saves the inputs and outputs of every
           `ac_freq` layer.
        3. SELECTIVE_OP_ACTIVATION_CHECKPOINTING: applies activation checkpointing to all layers, but only
           saves the activations of certain operations. Usually these operations are compute intensive and
           their activations are saved and not recomputed in the backward pass. All the remaining operations
           are recomputed in the backward pass.

        Args:
            ac_variant (ActivationCheckpointingVariants): The activation checkpointing variant to use.
            layers_fqn (str): The fully qualified name (FQN) of the layers to apply activation checkpointing to.
            model (nn.Module): The model to apply activation checkpointing to (in place).
            ac_fun_params (ActivationCheckpointedModelConfig.*Params): The parameters for the activation
                 checkpointing function.

        Raises:
            ValueError: If the activation checkpointing variant is not recognized or if the layers_fqn does not
                reference a ModuleList.
        """

        if ac_variant == ActivationCheckpointingVariants.FULL_ACTIVATION_CHECKPOINTING:
            apply_ac_fun = ActivationCheckpointing._apply_full_ac
        elif ac_variant == ActivationCheckpointingVariants.SELECTIVE_LAYER_ACTIVATION_CHECKPOINTING:
            apply_ac_fun = partial(
                ActivationCheckpointing._apply_selective_layer_ac,
                ac_freq=ac_fun_params.ac_freq,
            )
        elif ac_variant == ActivationCheckpointingVariants.SELECTIVE_OP_ACTIVATION_CHECKPOINTING:
            if len(ac_fun_params.save_ops_keys) > 0:
                apply_ac_fun = partial(
                    ActivationCheckpointing._apply_selective_op_ac, save_ops_keys=ac_fun_params.save_ops_keys
                )
            else:
                raise ValueError("No save_ops_keys provided for selective op activation checkpointing.")

        else:
            raise ValueError(f"Unknown activation checkpointing variant: {ac_variant}")

        layers = model.get_submodule(layers_fqn)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError(f"layers_fqn {layers_fqn} does not reference a ModuleList")

        print_rank_0(f"Applying activation checkpointing to {len(list(layers.named_children()))} layers...")

        for layer_id, transformer_block in layers.named_children():
            print_rank_0(f"Applying activation checkpointing to {layer_id}...")
            module_aced = apply_ac_fun(transformer_block)
            # the module_aced wraps the transformer_block as a CheckpointWrapper object.
            # module_aced._checkpoint_wrapped_module references the original transformer_block
            # we need to replace the original transformer_block with the wrapped one
            layers.register_module(layer_id, module_aced)

    @staticmethod
    def _apply_full_ac(module: nn.Module) -> nn.Module:
        module_aced = ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        return module_aced

    @staticmethod
    def _apply_selective_op_ac(module: nn.Module, save_ops_keys: list[str]) -> nn.Module:
        def _get_custom_policy(meta: dict[str, int], save_ops_set: Set):  # closure to capture meta
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops in save_ops_set, except every second mm
                # NOTE: we should make this configurable and not hide it in the code
                # To make this completely configurable, we would have to store the checkpointing frequency of every OP.
                to_save = func in save_ops_set and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

            return _custom_policy

        def _selective_checkpointing_context_fn():
            meta = defaultdict(int)
            save_ops_set = {ActivationCheckpointing.SAVE_DICT[key] for key in save_ops_keys}

            policy = _get_custom_policy(meta=meta, save_ops_set=save_ops_set)
            return create_selective_checkpoint_contexts(policy_fn_or_list=policy)

        module_saced = ptd_checkpoint_wrapper(
            module,
            context_fn=_selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
        return module_saced

    @staticmethod
    def _apply_selective_layer_ac(module: nn.Module, ac_freq: int) -> nn.Module:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if ptd_checkpoint_wrapper._count % ac_freq == 0:
            # we checkpoint activations every `ac_freq` layers
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            # in the other cases, we have to recompute the activations
            return module
