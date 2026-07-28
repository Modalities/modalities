"""End-to-end test of the YAML -> registry -> model path.

This is the integration test that proves the layer-spec design actually works through Modalities'
component factory: nested ``nemotron_layer_spec`` components must be built as *builders*, and the
model must then turn them into one independent module per layer position.
"""

from pathlib import Path

import pytest
import torch
from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.pydantic_if_types import PydanticPytorchModuleType
from modalities.models.nemotron.nemotron_layers import (
    Mamba2Layer,
    NemotronAttentionLayer,
    NemotronMLPLayer,
    NemotronMoELayer,
)
from modalities.models.nemotron.nemotron_model import NemotronLLM
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry

CONFIG_PATH = Path(__file__).parents[2] / "test_yaml_configs" / "nemotron_config_initialization.yaml"


class _RawModelConfig(BaseModel):
    model_raw: PydanticPytorchModuleType


def _build_raw_model() -> NemotronLLM:
    config_dict = load_app_config_dict(config_file_path=CONFIG_PATH)
    # The initialization sub-config carries placeholders that only the FSDP tests replace; the raw
    # model does not depend on them.
    config_dict.pop("model", None)
    component_factory = ComponentFactory(registry=Registry(COMPONENTS))
    components = component_factory.build_components(config_dict=config_dict, components_model_type=_RawModelConfig)
    return components.model_raw


@pytest.fixture(scope="module")
def model() -> NemotronLLM:
    return _build_raw_model()


def test_model_is_built_from_yaml(model):
    assert isinstance(model, NemotronLLM)
    assert model.n_layer == 8
    assert model.layer_pattern == "MEMEM*E-"


def test_layer_types_follow_the_configured_pattern(model):
    expected = [
        Mamba2Layer,
        NemotronMoELayer,
        Mamba2Layer,
        NemotronMoELayer,
        Mamba2Layer,
        NemotronAttentionLayer,
        NemotronMoELayer,
        NemotronMLPLayer,
    ]
    for layer_idx, expected_type in enumerate(expected):
        assert isinstance(model.transformer.h[str(layer_idx)], expected_type), f"layer {layer_idx}"


def test_repeated_layer_types_have_independent_weights(model):
    # The whole reason layer specs are builders: the component factory memoises components, so
    # injecting instantiated layers would make all three Mamba layers share one weight tensor.
    mamba_layers = [model.transformer.h[idx] for idx in ("0", "2", "4")]
    weights = [layer.mixer.in_proj.weight for layer in mamba_layers]
    assert len({id(w) for w in weights}) == 3
    assert not torch.equal(weights[0], weights[1])
    assert not torch.equal(weights[1], weights[2])

    moe_layers = [model.transformer.h[idx] for idx in ("1", "3", "6")]
    router_weights = [layer.moe.router.gate.weight for layer in moe_layers]
    assert len({id(w) for w in router_weights}) == 3


def test_component_hyperparameters_are_taken_from_yaml(model):
    mixer = model.transformer.h["0"].mixer
    assert (mixer.n_heads, mixer.head_dim, mixer.state_dim, mixer.n_groups) == (8, 32, 16, 2)
    assert mixer.d_inner == 8 * 32
    assert mixer.chunk_size == 32

    moe = model.transformer.h["1"].moe
    assert moe.router.num_experts == 8
    assert moe.router.top_k == 2
    assert moe.router.route_scale == 2.5
    assert moe.router.expert_bias is not None
    assert moe.experts.ffn_hidden == 64
    # 2 shared experts fused into one MLP of 2 * moe_ffn_hidden hidden units.
    assert moe.shared_experts.c_fc.out_features == 128
    assert moe.aux_loss_coeff == pytest.approx(1e-4)

    attn = model.transformer.h["5"].attn
    assert (attn.n_head_q, attn.n_head_kv, attn.head_dim) == (8, 2, 32)

    mlp = model.transformer.h["7"].mlp
    assert mlp.c_fc.out_features == 128


def test_forward_pass_through_the_configured_model(model):
    inputs = {"input_ids": torch.randint(0, model.vocab_size, (2, 16))}
    out = model(inputs)
    assert list(out.keys()) == ["logits", "moe_aux_loss"]
    assert out["logits"].shape == (2, 16, model.vocab_size)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["moe_aux_loss"])


def test_backward_pass_through_the_configured_model(model):
    out = model({"input_ids": torch.randint(0, model.vocab_size, (2, 16))})
    (out["logits"].float().mean() + out["moe_aux_loss"]).backward()
    assert model.transformer.wte.weight.grad is not None
    assert model.transformer.h["1"].moe.router.gate.weight.grad is not None
    model.zero_grad(set_to_none=True)


def test_activation_checkpointing_applies_to_the_layer_stack(model):
    # The Modalities activation checkpointing component requires transformer.h to be a ModuleDict
    # and addresses it by fully qualified name.
    from modalities.config.config import ActivationCheckpointedModelConfig
    from modalities.models.model_factory import ModelFactory
    from modalities.training.activation_checkpointing.activation_checkpointing_variants import (
        ActivationCheckpointingVariants,
    )

    checkpointed = ModelFactory.get_activation_checkpointed_fsdp2_model_(
        ac_variant=ActivationCheckpointingVariants.FULL_ACTIVATION_CHECKPOINTING,
        layers_fqn="transformer.h",
        model=_build_raw_model(),
        ac_fun_params=ActivationCheckpointedModelConfig.FullACParams(),
    )
    out = checkpointed({"input_ids": torch.randint(0, model.vocab_size, (1, 8))})
    assert out["logits"].shape == (1, 8, model.vocab_size)


def test_fsdp2_block_names_resolve_to_the_layer_classes(model):
    from modalities.util import get_module_class_from_name

    for block_name in ("Mamba2Layer", "NemotronMoELayer", "NemotronAttentionLayer", "NemotronMLPLayer"):
        assert get_module_class_from_name(model, block_name) is not None, block_name
