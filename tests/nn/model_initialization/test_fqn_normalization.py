"""Tests that initialization filters still match when the model is wrapped.

Activation checkpointing, torch.compile and FSDP1 all insert segments into a parameter's fully
qualified name. The initialization filters are written against the plain model, so those segments
have to be stripped before matching. Without that, applying activation checkpointing before weight
initialization makes every per-layer regex fail and the model silently keeps its default
initialization - it trains, just not as configured.
"""

import pytest
import torch
import torch.nn as nn

from modalities.nn.model_initialization.initialization_routines import InitializationRoutines, normalize_parameter_name
from modalities.nn.model_initialization.parameter_name_filters import RegexFilter


@pytest.mark.parametrize(
    "wrapped,plain",
    [
        ("transformer.h.0._checkpoint_wrapped_module.attn.c_proj.weight", "transformer.h.0.attn.c_proj.weight"),
        ("_orig_mod.transformer.wte.weight", "transformer.wte.weight"),
        ("_fsdp_wrapped_module.transformer.lm_head.weight", "transformer.lm_head.weight"),
        # Several wrappers can be layered (compile over activation checkpointing over FSDP1).
        (
            "_orig_mod.transformer.h.3._checkpoint_wrapped_module.mlp.W.weight",
            "transformer.h.3.mlp.W.weight",
        ),
        # An unwrapped name must pass through untouched.
        ("transformer.h.1.attn.q_attn.weight", "transformer.h.1.attn.q_attn.weight"),
    ],
)
def test_normalize_parameter_name_strips_wrapper_segments(wrapped, plain):
    assert normalize_parameter_name(wrapped) == plain


class _WrappedBlock(nn.Module):
    """Mimics how activation checkpointing renames a submodule's parameters."""

    def __init__(self):
        super().__init__()
        self._checkpoint_wrapped_module = nn.Linear(64, 64, bias=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict({"h": nn.ModuleDict({"0": _WrappedBlock()})})

    @property
    def wrapped_weight(self) -> nn.Parameter:
        return self.transformer["h"]["0"]._checkpoint_wrapped_module.weight


def test_initializer_matches_parameters_behind_a_wrapper():
    model = _Model()
    # The filter is written against the plain FQN, which is the whole point.
    regex_filter = RegexFilter(weights=[r"transformer\.h\.\d+\.weight"])
    assert any("_checkpoint_wrapped_module" in name for name, _ in model.named_parameters())

    std = 0.02
    initializer = InitializationRoutines.get_plain_initialization(
        mean=0.0, std=std, parameter_name_regexes=regex_filter, seed=42
    )
    with torch.no_grad():
        model.wrapped_weight.fill_(123.0)
    initializer.initialize_in_place(model)

    assert model.wrapped_weight.std().item() == pytest.approx(std, rel=0.1)
    assert model.wrapped_weight.abs().max().item() < 1.0, "parameter was not re-initialized"
