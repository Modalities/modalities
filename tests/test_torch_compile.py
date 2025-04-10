import pytest
import torch.nn as nn
from modalities.models.model_factory import ModelFactory
from modalities.models.gpt2.gpt2_model import (
    GPT2Block, GPT2LLM, AttentionConfig, AttentionImplementation, PositionTypes, 
    LayerNormWrapperConfig, LayerNorms, ActivationType, RotaryTransform
)

def create_gpt2_configs():
    attention_config = AttentionConfig(
        qkv_transforms=[
            AttentionConfig.QueryKeyValueTransformConfig(
                type_hint="RotaryTransform",
                config=AttentionConfig.QueryKeyValueTransformConfig.RotaryTransformConfig(
                    n_embd=512,
                    n_head=8,
                    seq_length_dim=-2,
                    base_freq=10000
                )
            )
        ]
    )
    norm_config = LayerNormWrapperConfig(
        norm_type=LayerNorms.layer_norm,
        config={"normalized_shape": 512}
    )
    return attention_config, norm_config

@pytest.fixture
def gpt2_model():
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
        use_weight_tying=False
    )
    return model

def test_get_compiled_model_compiles_blocks(gpt2_model):
    original_blocks = list(gpt2_model.transformer.h)
    original_wte = gpt2_model.transformer.wte
    original_lm_head = gpt2_model.transformer.lm_head

    block_names = ["GPT2Block"]
    result_model = ModelFactory.get_compiled_model(gpt2_model, block_names)

    assert len(result_model.transformer.h) == 4, "Should still have four blocks"
    for i, (original_block, new_block) in enumerate(zip(original_blocks, result_model.transformer.h)):
        assert new_block is not original_block, f"Block {i} should be a compiled version"
        assert isinstance(new_block, nn.Module), f"Block {i} should be an nn.Module"
    assert result_model.transformer.wte is original_wte, "Embedding layer should remain unchanged"
    assert result_model.transformer.lm_head is original_lm_head, "LM head should remain unchanged"
    assert result_model is gpt2_model, "Should return the same model instance"

def test_get_compiled_model_invalid_block_name(gpt2_model):
    """
    Test that get_compiled_model does nothing when given an invalid block name.
    """
    original_model_dict = dict(gpt2_model.named_modules())
    block_names = ["InvalidBlock"]
    result_model = ModelFactory.get_compiled_model(gpt2_model, block_names)

    new_model_dict = dict(result_model.named_modules())
    assert new_model_dict == original_model_dict, "Model should remain unchanged with invalid block name"

def test_get_compiled_model_no_matching_blocks(gpt2_model):
    """
    Test that get_compiled_model does nothing if no blocks match the specified types.
    """
    original_model_dict = dict(gpt2_model.named_modules())
    block_names = ["Conv2d"]
    result_model = ModelFactory.get_compiled_model(gpt2_model, block_names)

    new_model_dict = dict(result_model.named_modules())
    assert new_model_dict == original_model_dict, "Model should remain unchanged when no blocks match"

def test_get_compiled_model_empty_block_names(gpt2_model):
    original_model_dict = dict(gpt2_model.named_modules())
    result_model = ModelFactory.get_compiled_model(gpt2_model, block_names=[])

    new_model_dict = dict(result_model.named_modules())
    assert new_model_dict == original_model_dict, "Model should remain unchanged with empty block_names"
    assert result_model is gpt2_model, "Should return the same model instance"