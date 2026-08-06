import pytest
import torch
import torch.nn as nn

from modalities.models.components.layer_norms import LayerNormConfig
from modalities.models.gpt2.gpt2_model import (
    GPT2LLM,
    AttentionConfig,
    AttentionImplementation,
    LayerNorms,
    LayerNormWrapperConfig,
    PositionTypes,
)
from modalities.models.model import ActivationType

VOCAB_SIZE = 1000
EMBEDDING_DIM = 64


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def create_gpt2_model(use_weight_tying: bool) -> GPT2LLM:
    vocab_size = VOCAB_SIZE
    n_embd = EMBEDDING_DIM
    sequence_length = 128
    n_layer = 2
    n_head_q = 4
    n_head_kv = 2
    ffn_hidden = 256
    dropout = 0.1
    bias = True
    poe_type = PositionTypes.NOPE
    activation_type = ActivationType.GELU
    attention_implementation = AttentionImplementation.PYTORCH_FLASH
    attention_config = AttentionConfig(
        qkv_transforms=[
            AttentionConfig.QueryKeyValueTransformConfig(
                type_hint="RotaryTransform",
                config=AttentionConfig.QueryKeyValueTransformConfig.RotaryTransformConfig(
                    n_embd=n_embd,
                    n_head=n_head_q,
                    seq_length_dim=-2,
                    base_freq=10000,
                ),
            )
        ]
    )
    attention_norm_config = LayerNormWrapperConfig(
        norm_type=LayerNorms.layer_norm, config=LayerNormConfig(normalized_shape=n_embd)
    )
    ffn_norm_config = LayerNormWrapperConfig(
        norm_type=LayerNorms.layer_norm, config=LayerNormConfig(normalized_shape=n_embd)
    )
    lm_head_norm_config = LayerNormWrapperConfig(
        norm_type=LayerNorms.layer_norm, config=LayerNormConfig(normalized_shape=n_embd)
    )

    return GPT2LLM(
        sample_key="input_ids",
        prediction_key="logits",
        poe_type=poe_type,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head_q=n_head_q,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        ffn_hidden=ffn_hidden,
        dropout=dropout,
        bias=bias,
        activation_type=activation_type,
        attention_implementation=attention_implementation,
        attention_config=attention_config,
        attention_norm_config=attention_norm_config,
        ffn_norm_config=ffn_norm_config,
        lm_head_norm_config=lm_head_norm_config,
        use_weight_tying=use_weight_tying,
    )


@pytest.mark.parametrize("use_weight_tying", [True, False])
def test_weight_tying_behavior(use_weight_tying):
    model = create_gpt2_model(use_weight_tying)
    if use_weight_tying:
        assert (
            model.transformer.wte.weight is model.transformer.lm_head.weight
        ), "Weight tying failed: Embedding and LM head weights are not the same."
    else:
        assert (
            model.transformer.wte.weight is not model.transformer.lm_head.weight
        ), "Weight tying failed: Embedding and LM head weights should be different."


def test_absolute_embeddings_use_explicit_global_position_ids():
    model = create_gpt2_model(use_weight_tying=False)
    model.poe_type = PositionTypes.ABSOLUTE
    model.transformer.wte = nn.Embedding(1, EMBEDDING_DIM)
    model.transformer.wpe = nn.Embedding(8, EMBEDDING_DIM)
    model.transformer.drop = nn.Identity()
    model.transformer.h = nn.ModuleDict()
    model.transformer.lm_head_norm = nn.Identity()
    model.transformer.lm_head = nn.Identity()
    nn.init.zeros_(model.transformer.wte.weight)
    with torch.no_grad():
        model.transformer.wpe.weight.copy_(torch.arange(8).unsqueeze(1).expand(-1, EMBEDDING_DIM))

    output = model.forward_impl(torch.zeros(1, 2, dtype=torch.long), position_ids=torch.tensor([[5, 7]]))

    torch.testing.assert_close(output[0, :, 0], torch.tensor([5.0, 7.0]))


def test_pipeline_stage_forwards_position_ids_with_hidden_states():
    model = create_gpt2_model(use_weight_tying=False)
    model.transformer.h = nn.ModuleDict()
    del model.transformer["lm_head"]
    position_ids = torch.tensor([[4, 5]])

    hidden_states, forwarded_position_ids = model.forward_impl(
        torch.zeros(1, 2, dtype=torch.long), position_ids=position_ids
    )

    assert hidden_states.shape == (1, 2, EMBEDDING_DIM)
    assert forwarded_position_ids is position_ids


def test_weight_tying_parameter_count():
    model_with_tying = create_gpt2_model(use_weight_tying=True)
    param_count_tied = count_parameters(model_with_tying)
    print(f"Parameter count with weight tying: {param_count_tied}")

    model_without_tying = create_gpt2_model(use_weight_tying=False)
    param_count_not_tied = count_parameters(model_without_tying)
    expected_difference = VOCAB_SIZE * EMBEDDING_DIM
    assert (
        param_count_not_tied == param_count_tied + expected_difference
    ), "Parameter count mismatch when using weight tying."


@pytest.mark.parametrize("use_weight_tying", [True, False])
def test_weight_tying_named_parameters(use_weight_tying):
    model = create_gpt2_model(use_weight_tying)
    named_params = [name for name, _ in model.named_parameters()]

    if use_weight_tying:
        assert (
            "transformer.lm_head.weight" not in named_params
        ), "transformer.lm_head.weight should not appear in named_parameters when weight tying is used."
    else:
        assert (
            "transformer.lm_head.weight" in named_params
        ), "transformer.lm_head.weight should appear in named_parameters when weight tying is not used."
