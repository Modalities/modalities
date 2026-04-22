import copy
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

from modalities.models.components.layer_norms import LayerNormConfig
from modalities.models.gpt2.gpt2_model import (
    GPT2LLM,
    ActivationType,
    AttentionConfig,
    AttentionImplementation,
    LayerNorms,
    LayerNormWrapperConfig,
    PositionTypes,
    QueryKeyValueTransformType,
    is_flash_attn_v4_available,
)
from modalities.models.model_factory import ModelFactory


def create_gpt2_configs() -> tuple[AttentionConfig, LayerNormWrapperConfig]:
    attention_config = AttentionConfig(
        qkv_transforms=[
            AttentionConfig.QueryKeyValueTransformConfig(
                type_hint=cast(Any, QueryKeyValueTransformType.RotaryTransform.name),
                config=AttentionConfig.QueryKeyValueTransformConfig.RotaryTransformConfig(
                    n_embd=512, n_head=8, seq_length_dim=-2, base_freq=10000
                ),
            )
        ]
    )
    norm_config = LayerNormWrapperConfig(
        norm_type=LayerNorms.layer_norm,
        config=LayerNormConfig(normalized_shape=512, eps=1e-6, elementwise_affine=True, bias=True),
    )
    return attention_config, norm_config


@pytest.fixture
def gpt2_model() -> GPT2LLM:
    attention_config, norm_config = create_gpt2_configs()
    model = GPT2LLM(
        sample_key="input_ids",
        prediction_key="logits",
        poe_type=PositionTypes.NOPE,
        sequence_length=256,
        vocab_size=1024,
        n_layer=4,
        n_head_q=8,
        n_head_kv=4,
        n_embd=512,
        ffn_hidden=2048,
        dropout=0.1,
        bias=True,
        activation_type=ActivationType.SWIGLU,
        attention_implementation=AttentionImplementation.PYTORCH_FLASH,
        attention_config=attention_config,
        attention_norm_config=norm_config,
        ffn_norm_config=norm_config,
        lm_head_norm_config=norm_config,
        use_weight_tying=True,
    )
    return model


def test_get_compiled_model_compiles_blocks(gpt2_model: GPT2LLM) -> None:
    original_model = copy.deepcopy(gpt2_model)
    original_wte = gpt2_model.transformer.wte
    original_lm_head = gpt2_model.transformer.lm_head

    block_names = ["GPT2Block"]
    result_model = ModelFactory.get_compiled_model(gpt2_model, block_names, fullgraph=True)

    assert len(result_model.transformer.h) == 4, "Should still have four blocks"
    for i, (original_block_idx, new_block_idx) in enumerate(
        zip(original_model.transformer.h, result_model.transformer.h)
    ):
        assert (
            result_model.transformer.h[new_block_idx] is not original_model.transformer.h[original_block_idx]
        ), f"Block {i} should be a compiled version"
        assert isinstance(result_model.transformer.h[new_block_idx], nn.Module), f"Block {i} should be an nn.Module"
    assert result_model.transformer.wte is original_wte, "Embedding layer should remain unchanged"
    assert result_model.transformer.lm_head is original_lm_head, "LM head should remain unchanged"
    assert result_model is gpt2_model, "Should return the same model instance"


def test_get_compiled_model_no_matching_blocks(gpt2_model: GPT2LLM) -> None:
    """
    Test that get_compiled_model raises a ValueError if no blocks match the specified types.
    """
    block_name = "Conv2d"
    with pytest.raises(ValueError, match=f"The block name {block_name} does not match any modules in the model"):
        ModelFactory.get_compiled_model(gpt2_model, block_names=[block_name], fullgraph=True)


def test_get_compiled_model_empty_block_names(gpt2_model: GPT2LLM) -> None:
    original_model_dict = dict(gpt2_model.named_modules())
    result_model = ModelFactory.get_compiled_model(gpt2_model, block_names=[], fullgraph=True)

    new_model_dict = dict(result_model.named_modules())
    assert new_model_dict == original_model_dict, "Model should remain unchanged with empty block_names"
    assert result_model is gpt2_model, "Should return the same model instance"


@pytest.mark.skipif(not is_flash_attn_v4_available(), reason="FA4 not installed")
def test_get_compiled_model_disables_fullgraph_for_fa4(monkeypatch: MonkeyPatch, gpt2_model: GPT2LLM) -> None:
    recorded_fullgraph_values: list[bool] = []

    for block in gpt2_model.transformer.h.values():
        block.attn.attention_impl = AttentionImplementation.DAO_FLASH_V4

    def fake_compile(module: nn.Module, fullgraph: bool, options: dict[str, object]) -> nn.Module:
        recorded_fullgraph_values.append(fullgraph)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)

    ModelFactory.get_compiled_model(gpt2_model, ["GPT2Block"], fullgraph=True)

    assert recorded_fullgraph_values == [False] * len(gpt2_model.transformer.h)
