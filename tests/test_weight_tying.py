import math

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from torch.distributed.device_mesh import DeviceMesh

from modalities.config.config import GPT2ModelTPConfig
from modalities.models.components.layer_norms import LayerNormConfig, PytorchRMSLayerNormConfig
from modalities.models.gpt2.gpt2_model import (
    GPT2LLM,
    AttentionConfig,
    AttentionImplementation,
    LayerNorms,
    LayerNormWrapperConfig,
    PositionTypes,
)
from modalities.models.gpt2.llama3_like_initialization import Llama3Initializer
from modalities.models.model import ActivationType
from modalities.models.parallelism.pipeline_parallelism_configs import StagedPipelineConfig
from modalities.models.parallelism.stages_generator import GPT2LLMStagesGenerator
from modalities.models.weight_tying import has_tied_word_embeddings
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees

VOCAB_SIZE = 1000
EMBEDDING_DIM = 64


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def create_gpt2_model(
    use_weight_tying: bool,
    activation_type: ActivationType = ActivationType.GELU,
    bias: bool = True,
    norm_type: LayerNorms = LayerNorms.layer_norm,
) -> GPT2LLM:
    vocab_size = VOCAB_SIZE
    n_embd = EMBEDDING_DIM
    sequence_length = 128
    n_layer = 2
    n_head_q = 4
    n_head_kv = 2
    ffn_hidden = 256
    dropout = 0.1
    poe_type = PositionTypes.NOPE
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

    def _make_norm_config() -> LayerNormWrapperConfig:
        if norm_type == LayerNorms.pytorch_rms_norm:
            return LayerNormWrapperConfig(
                norm_type=norm_type, config=PytorchRMSLayerNormConfig(normalized_shape=n_embd)
            )
        return LayerNormWrapperConfig(norm_type=norm_type, config=LayerNormConfig(normalized_shape=n_embd))

    attention_norm_config = _make_norm_config()
    ffn_norm_config = _make_norm_config()
    lm_head_norm_config = _make_norm_config()

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


def create_device_mesh_stub(*mesh_dim_names: str) -> DeviceMesh:
    device_mesh = DeviceMesh.__new__(DeviceMesh)
    device_mesh.mesh_dim_names = mesh_dim_names
    return device_mesh


@pytest.mark.parametrize("use_weight_tying", [True, False])
def test_weight_tying_behavior(use_weight_tying):
    model = create_gpt2_model(use_weight_tying)
    assert model.has_tied_word_embeddings is use_weight_tying

    if use_weight_tying:
        assert (
            model.transformer.wte.weight is model.transformer.lm_head.weight
        ), "Weight tying failed: Embedding and LM head weights are not the same."
    else:
        assert (
            model.transformer.wte.weight is not model.transformer.lm_head.weight
        ), "Weight tying failed: Embedding and LM head weights should be different."


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


def test_has_tied_word_embeddings_requires_model_capability():
    with pytest.raises(TypeError, match="must define 'has_tied_word_embeddings'"):
        has_tied_word_embeddings(nn.Linear(1, 1))


@pytest.mark.parametrize("module_name", ["transformer", "wte", "lm_head"])
def test_has_tied_word_embeddings_handles_pipeline_stage(module_name: str):
    model = create_gpt2_model(use_weight_tying=True)
    if module_name == "transformer":
        del model.transformer
    else:
        del model.transformer[module_name]

    assert has_tied_word_embeddings(model) is False


def test_tp_config_rejects_tied_word_embeddings():
    model = create_gpt2_model(use_weight_tying=True)
    device_mesh = create_device_mesh_stub(ParallelismDegrees.TP.value)

    with pytest.raises(ValidationError, match="Tied word embeddings are not supported with Tensor Parallelism"):
        GPT2ModelTPConfig(model=model, device_mesh=device_mesh)


@pytest.mark.parametrize("use_weight_tying", [True, False])
def test_llama3_init_keeps_output_projection_small(use_weight_tying: bool):
    """Regression test for the weight-tying init bug.

    With weight tying, ``transformer.wte.weight`` *is* the output projection
    (``lm_head`` shares the same tensor), so it must be initialized with the small
    output std ``1 / sqrt(n_embd)`` -- not the embedding std of 1. Otherwise the tied
    matrix produces logits ~sqrt(n_embd)x too large at init and the loss/grad norm
    explode (observed: initial loss ~1685 instead of ~ln(vocab_size)).
    """
    n_embd = EMBEDDING_DIM
    expected_output_std = 1 / math.sqrt(n_embd)

    # SwiGLU + RMSNorm + no bias so the Llama3Initializer's FQN regexes fully match
    # the model and it rejects no parameters.
    model = create_gpt2_model(
        use_weight_tying=use_weight_tying,
        activation_type=ActivationType.SWIGLU,
        bias=False,
        norm_type=LayerNorms.pytorch_rms_norm,
    )
    initializer = Llama3Initializer(num_layers=2, n_embd=n_embd, depth_init=True, use_weight_tying=use_weight_tying)
    # Mirror the production flow (model_factory applies the initializer under no_grad).
    with torch.no_grad():
        initializer.initialize_in_place(model)

    # The logit-producing matrix must be small regardless of weight tying.
    output_proj_std = model.transformer.lm_head.weight.detach().float().std().item()
    assert output_proj_std == pytest.approx(expected_output_std, rel=0.15)

    if use_weight_tying:
        # Tied: embedding and output projection are the same (small) tensor.
        assert model.transformer.wte.weight is model.transformer.lm_head.weight
    else:
        # Untied: the embedding keeps the Llama3/TorchTitan std of 1.
        embedding_std = model.transformer.wte.weight.detach().float().std().item()
        assert embedding_std == pytest.approx(1.0, rel=0.15)


def test_tp_config_allows_untied_word_embeddings():
    model = create_gpt2_model(use_weight_tying=False)
    device_mesh = create_device_mesh_stub(ParallelismDegrees.TP.value)

    GPT2ModelTPConfig(model=model, device_mesh=device_mesh)


def test_pp_config_rejects_tied_word_embeddings():
    model = create_gpt2_model(use_weight_tying=True)
    device_mesh = create_device_mesh_stub(ParallelismDegrees.PP.value)

    with pytest.raises(ValidationError, match="Tied word embeddings are not supported with Pipeline Parallelism"):
        StagedPipelineConfig(
            whole_model=model,
            stages_generator=GPT2LLMStagesGenerator(num_model_layers=model.n_layer),
            device_mesh=device_mesh,
            local_rank=0,
            pp_schedule_name="gpipe",
            num_layers_per_stage=1,
        )


def test_pp_config_allows_untied_word_embeddings():
    model = create_gpt2_model(use_weight_tying=False)
    device_mesh = create_device_mesh_stub(ParallelismDegrees.PP.value)

    StagedPipelineConfig(
        whole_model=model,
        stages_generator=GPT2LLMStagesGenerator(num_model_layers=model.n_layer),
        device_mesh=device_mesh,
        local_rank=0,
        pp_schedule_name="gpipe",
        num_layers_per_stage=1,
    )
