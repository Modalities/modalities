import torch

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
from modalities.models.model_factory import GPT2ModelFactory

VOCAB_SIZE = 1000
EMBEDDING_DIM = 64
SEQUENCE_LENGTH = 32
BATCH_SIZE = 2


def create_gpt2_model(return_hidden_states: bool = False) -> GPT2LLM:
    n_embd = EMBEDDING_DIM
    n_head_q = 4
    norm_config = LayerNormWrapperConfig(norm_type=LayerNorms.layer_norm, config=LayerNormConfig(normalized_shape=n_embd))
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
    return GPT2LLM(
        sample_key="input_ids",
        prediction_key="logits",
        poe_type=PositionTypes.NOPE,
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_head_q=n_head_q,
        n_head_kv=2,
        n_embd=n_embd,
        ffn_hidden=256,
        dropout=0.0,
        bias=True,
        activation_type=ActivationType.GELU,
        attention_implementation=AttentionImplementation.PYTORCH_FLASH,
        attention_config=attention_config,
        attention_norm_config=norm_config,
        ffn_norm_config=norm_config,
        lm_head_norm_config=norm_config,
        use_weight_tying=False,
        return_hidden_states=return_hidden_states,
    )


def _sample_inputs() -> dict[str, torch.Tensor]:
    return {"input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH))}


def test_return_hidden_states_default_is_false():
    model = create_gpt2_model()
    assert model.return_hidden_states is False


def test_return_hidden_states_changes_output_shape():
    inputs = _sample_inputs()

    logits = create_gpt2_model(return_hidden_states=False)(inputs)["logits"]
    assert logits.shape == (BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE)

    hidden_states = create_gpt2_model(return_hidden_states=True)(inputs)["logits"]
    assert hidden_states.shape == (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM)


def test_return_hidden_states_matches_full_forward_lm_head_projection():
    """The returned hidden states must be exactly the lm_head's input: applying lm_head to
    them must reproduce the same logits as the default (return_hidden_states=False) forward,
    since ChunkedLMHeadCrossEntropyLoss relies on this equivalence to apply the lm_head
    chunk-wise outside the model instead of inside forward_impl.
    """
    inputs = _sample_inputs()

    model_logits = create_gpt2_model(return_hidden_states=False)
    model_hidden = create_gpt2_model(return_hidden_states=True)
    model_hidden.load_state_dict(model_logits.state_dict())

    with torch.no_grad():
        logits = model_logits(inputs)["logits"]
        hidden_states = model_hidden(inputs)["logits"]
        projected_logits = model_hidden.transformer.lm_head(hidden_states)

    torch.testing.assert_close(projected_logits, logits)


def test_get_gpt2_model_factory_passes_through_return_hidden_states():
    n_embd = EMBEDDING_DIM
    n_head_q = 4
    norm_config = LayerNormWrapperConfig(norm_type=LayerNorms.layer_norm, config=LayerNormConfig(normalized_shape=n_embd))
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
    model = GPT2ModelFactory.get_gpt2_model(
        sample_key="input_ids",
        prediction_key="logits",
        poe_type=PositionTypes.NOPE,
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_head_q=n_head_q,
        n_head_kv=2,
        n_embd=n_embd,
        ffn_hidden=256,
        dropout=0.0,
        bias=True,
        activation_type=ActivationType.GELU,
        attention_implementation=AttentionImplementation.PYTORCH_FLASH,
        attention_config=attention_config,
        attention_norm_config=norm_config,
        ffn_norm_config=norm_config,
        lm_head_norm_config=norm_config,
        use_weight_tying=False,
        return_hidden_states=True,
    )
    assert model.return_hidden_states is True
    logits = model(_sample_inputs())["logits"]
    assert logits.shape == (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM)
