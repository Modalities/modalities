from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, apply_activation_checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP1
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from modalities.config.config import SelectiveActivationCheckpointedModelConfig
from modalities.training.activation_checkpointing.activation_checkpointing_variants import (
    SelectiveActivationCheckpointingVariants,
)
from modalities.util import get_module_class_from_name, print_rank_0


def is_module_to_apply_activation_checkpointing(
    submodule: nn.Module, activation_checkpointing_modules: list[type]
) -> bool:
    return isinstance(submodule, tuple(activation_checkpointing_modules))


def apply_activation_checkpointing_inplace(model: nn.Module, activation_checkpointing_modules: list[str]):
    activation_checkpointing_module_types = [
        get_module_class_from_name(model, m) for m in activation_checkpointing_modules
    ]
    if not isinstance(model, (FSDP1, FSDP2)):
        raise ValueError("activation checkpointing can only be applied to FSDP wrapped models!")
    non_reentrant_wrapper = partial(ptd_checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT, debug=False)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: is_module_to_apply_activation_checkpointing(
            submodule, activation_checkpointing_module_types
        ),
    )


class SelectiveActivationCheckpointing:
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

    @staticmethod
    def apply_selective_activation_checkpointing_(
        sac_variant: SelectiveActivationCheckpointingVariants,
        layers_fqn: str,
        model: nn.Module,
        sac_fun_params: (
            SelectiveActivationCheckpointedModelConfig.FullACParams
            | SelectiveActivationCheckpointedModelConfig.SelectiveLayerACParams
            | SelectiveActivationCheckpointedModelConfig.SelectiveOpACParams
        ),
    ):
        """
        Applies activation checkpointing to the gpt2 model. There are three variants of activation checkpointing:
        1. FULL_ACTIVATION_CHECKPOINTING: applies activation checkpointing to all layers
        2. SELECTIVE_LAYER_ACTIVATION_CHECKPOINTING: applies activation checkpointing to every `ac_freq` layer
        3. SELECTIVE_OP_ACTIVATION_CHECKPOINTING: applies activation checkpointing to all layers, but only saves the
        """
        if sac_variant == SelectiveActivationCheckpointingVariants.FULL_ACTIVATION_CHECKPOINTING:
            apply_ac_fun = SelectiveActivationCheckpointing._apply_full_ac
        elif sac_variant == SelectiveActivationCheckpointingVariants.SELECTIVE_LAYER_ACTIVATION_CHECKPOINTING:
            apply_ac_fun = partial(
                SelectiveActivationCheckpointing._apply_selective_layer_ac,
                ac_freq=sac_fun_params.ac_freq,
            )
        elif sac_variant == SelectiveActivationCheckpointingVariants.SELECTIVE_OP_ACTIVATION_CHECKPOINTING:
            apply_ac_fun = SelectiveActivationCheckpointing._apply_selective_op_ac
        else:
            raise ValueError(f"Unknown activation checkpointing variant: {sac_variant}")

        layers = model.get_submodule(layers_fqn)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError(f"layers_fqn {layers_fqn} does not reference a ModuleList")

        for layer_id, transformer_block in layers.named_children():
            print_rank_0(f"Applying activation checkpointing to {layer_id}...")
            module_saced = apply_ac_fun(transformer_block)
            # the module_saced wraps the transformer_block as a CheckpointWrapper object.
            # module_saced._checkpoint_wrapped_module references the original transformer_block
            # we need to replace the original transformer_block with the wrapped one
            layers.register_module(layer_id, module_saced)

    @staticmethod
    def _apply_full_ac(module: nn.Module) -> nn.Module:
        module_saced = ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        return module_saced

    @staticmethod
    def _apply_selective_op_ac(module: nn.Module) -> nn.Module:
        def _get_custom_policy(meta):  # closure to capture meta
            def _custom_policy(ctx, func, *args, **kwargs):
                # This is the list of ops that are saved by default in torch titan.
                # These operations are typically compute intensive and their activations are
                # therefore saved and not recomputed in the backward pass.
                # This list differs from the compute intensive ops list in the
                # pytorch AC tutorial: https://pytorch.org/blog/activation-checkpointing-techniques/
                # TODO: Optimize this list for our GP2 implementation!
                save_list = {  # default save list from torch titan
                    torch.ops.aten.mm.default,
                    torch.ops.aten._scaled_dot_product_efficient_attention.default,
                    torch.ops.aten._scaled_dot_product_flash_attention.default,
                    torch.ops._c10d_functional.reduce_scatter_tensor.default,
                    # for low precision training, it's useful to always save
                    # the result of max, since the absolute maximum is
                    # used to compute the scaling factor for quantization.
                    torch.ops.aten.max.default,
                }

                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in save_list and not (func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0)
                return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

            return _custom_policy

        def _selective_checkpointing_context_fn():
            meta = defaultdict(int)
            # For now, we only allow for a single AC policy
            # (i.e., the torch titan LLama 3 one) to be used
            policy = _get_custom_policy(meta)
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
