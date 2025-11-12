import torch

from modalities.models.gpt2.gpt2_model import (
    GPT2LLM,
    AttentionConfig,
    AttentionImplementation,
    LayerNormWrapperConfig,
    PositionTypes,
)
from modalities.models.model import ActivationType
from modalities.models.model_factory import ModelFactory
from modalities.nn.model_initialization.composed_initialization import ComposedInitializationRoutines
from modalities.nn.model_initialization.parameter_name_filters import SupportWeightInitModels, WeightInitTypes


def test_deferred_initialization_produces_same_weights_as_eager_initialization():
    with torch.device("cuda"):
        gpt2_model_eager = _build_gpt2_model()
    gpt2_model_eager = _apply_initialization(gpt2_model_eager)
    with torch.device("meta"):
        gpt2_model_deferred = _build_gpt2_model()
    gpt2_model_deferred = _apply_initialization(gpt2_model_deferred)

    # check that both models have the same parameters
    for (name_eager, param_eager), (name_deferred, param_deferred) in zip(
        gpt2_model_eager.named_parameters(),
        gpt2_model_deferred.named_parameters(),
    ):
        assert name_eager == name_deferred, f"Parameter names do not match: {name_eager} != {name_deferred}"
        assert torch.allclose(param_eager, param_deferred), f"Parameters do not match for {name_eager}"
    # check that both models have the same buffers
    for (name_eager, buffer_eager), (name_deferred, buffer_deferred) in zip(
        gpt2_model_eager.named_buffers(),
        gpt2_model_deferred.named_buffers(),
    ):
        assert name_eager == name_deferred, f"Buffer names do not match: {name_eager} != {name_deferred}"
        assert torch.allclose(buffer_eager, buffer_deferred), f"Buffers do not match for {name_eager}"


def _build_gpt2_model() -> GPT2LLM:
    """
    Eagerly builds a GPT2LLM instance using the exact values
    from tests/end2end_tests/configs/gpt2_train_num_steps_7_pp.yaml (model_raw section).
    """
    sequence_length = 256
    n_embd = 128
    n_layer = 2
    n_head_q = 8
    n_head_kv = 8
    ffn_hidden = 128

    # LayerNorm wrapper configs (mirroring yaml: eps=1e-5, normalized_shape = n_embd)
    ln_cfg = LayerNormWrapperConfig(
        norm_type="layer_norm",
        config={
            "normalized_shape": n_embd,
            "eps": 1e-5,
        },
    )

    attention_cfg = AttentionConfig(
        qkv_transforms=[
            {
                "type_hint": "RotaryTransform",
                "config": {
                    "n_embd": n_embd,
                    "n_head": n_head_q,  # matches yaml comment: must use n_head_q here
                    "seq_length_dim": -2,
                    "base_freq": 10000,
                },
            }
        ]
    )

    model = GPT2LLM(
        sample_key="input_ids",
        prediction_key="logits",
        poe_type=PositionTypes.NOPE,
        sequence_length=sequence_length,
        vocab_size=50304,
        n_layer=n_layer,
        n_head_q=n_head_q,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        ffn_hidden=ffn_hidden,
        dropout=0.0,
        bias=True,
        activation_type=ActivationType.SWIGLU,
        attention_implementation=AttentionImplementation.MANUAL,
        attention_config=attention_cfg,
        attention_norm_config=ln_cfg,
        ffn_norm_config=ln_cfg,
        lm_head_norm_config=ln_cfg,
        use_weight_tying=False,
        seed=42,
        enforce_swiglu_hidden_dim_multiple_of=256,
    )
    return model


def _apply_initialization(model: GPT2LLM) -> GPT2LLM:
    model_initializer = ComposedInitializationRoutines.get_composed_model_initializer(
        model_type=SupportWeightInitModels.GPT2,
        weight_init_type=WeightInitTypes.SCALED,
        mean=0.0,
        std=0.02,
        num_layers=2,
    )
    torch.manual_seed(42)
    return ModelFactory.get_weight_initialized_model(model, model_initializer)
