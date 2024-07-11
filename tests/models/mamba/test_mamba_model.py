import os

import pytest
import torch

from modalities.models.mamba.mamba_model import _init_weights, create_block


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_mixer_model_forward(batch_size, sequence_length, vocab_size, mixer_model, d_model):
    x = torch.randint(size=(batch_size, sequence_length), high=vocab_size).to("cuda")
    mixer_model = mixer_model.to("cuda")
    y = mixer_model(x)
    assert y.shape == (batch_size, sequence_length, d_model)
    assert y.shape != x.shape


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_mixer_model_allocate_inference_cache(batch_size, sequence_length, mixer_model, n_layer):
    mixer_model = mixer_model.to("cuda")
    computed_inference_cache = mixer_model.allocate_inference_cache(batch_size, sequence_length)
    assert len(computed_inference_cache) == n_layer


def test__init_weights(linear_layer, embedding_layer, n_layer):
    _init_weights(linear_layer, n_layer)
    assert int(linear_layer.bias.sum()) == 0

    embedding_layer_weights_before = embedding_layer.weight.clone().detach()
    _init_weights(embedding_layer, n_layer)
    embedding_layer_weights_after = embedding_layer.weight
    assert (embedding_layer_weights_before != embedding_layer_weights_after).any()


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_mamba_llm_forward(mamba_llm, batch_size, sequence_length, vocab_size, prediction_key):
    mamba_llm = mamba_llm.to("cuda")
    x = torch.randint(size=(batch_size, sequence_length), high=vocab_size).to("cuda")
    inputs = {"input_ids": x}
    y = mamba_llm(inputs)
    assert prediction_key in y.keys()
    assert y[prediction_key].shape == (batch_size, sequence_length, vocab_size)


def test__create_block(
    d_model, ssm_cfg, norm_epsilon, rms_norm, residual_in_fp32, fused_add_norm, layer_idx, device, dtype
):
    test_block = create_block(
        d_model=d_model,
        ssm_cfg=ssm_cfg,
        norm_epsilon=norm_epsilon,
        rms_norm=rms_norm,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        layer_idx=layer_idx,
        device=device,
        dtype=dtype,
    )
    assert test_block.norm.normalized_shape[0] == d_model
    assert test_block.mixer.d_model == d_model


def test_tie_weights(mamba_llm):
    assert (mamba_llm.lm_head.weight != mamba_llm.backbone.embedding.weight).any()
    mamba_llm.tie_embeddings = True
    mamba_llm.tie_weights()
    assert (mamba_llm.lm_head.weight == mamba_llm.backbone.embedding.weight).all()
