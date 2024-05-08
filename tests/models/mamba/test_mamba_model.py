import pytest
import torch
from torch import nn

from modalities.models.mamba.mamba_model import MixerModel, _init_weights, MambaLLM, create_block


@pytest.fixture()
def batch_size():
    return 2


@pytest.fixture()
def vocab_size():
    return 1024


@pytest.fixture()
def sequence_length():
    return 64


@pytest.fixture()
def n_layer():
    return 2


@pytest.fixture()
def d_model():
    return 12


@pytest.fixture()
def ssm_config():
    return {}


@pytest.fixture()
def mixer_model(d_model, n_layer, vocab_size):
    return MixerModel(d_model, n_layer, vocab_size)


@pytest.fixture()
def prediction_key():
    return "logits"


@pytest.fixture()
def sample_key():
    return "input_ids"


@pytest.fixture()
def mamba_llm(d_model, n_layer, vocab_size, prediction_key, sample_key):
    return MambaLLM(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size, ssm_cfg={}, rms_norm=True,
                    residual_in_fp32=False, fused_add_norm=False, pad_vocab_size_multiple=1, tie_embeddings=False,
                    prediction_key=prediction_key, sample_key=sample_key)


@pytest.fixture()
def linear_layer():
    return nn.Linear(in_features=16, out_features=24)


@pytest.fixture()
def embedding_layer(vocab_size, d_model):
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="We need cuda to run Mamba.")
def test_mixer_model_forward(batch_size, sequence_length, vocab_size, mixer_model, d_model):
    x = torch.randint(size=(batch_size, sequence_length), high=vocab_size).to("cuda")
    mixer_model = mixer_model.to("cuda")
    y = mixer_model(x)
    assert y.shape == (batch_size, sequence_length, d_model)
    assert y.shape != x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="We need cuda to run Mamba.")
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


def test_mamba_llm_forward(mamba_llm, batch_size, sequence_length, vocab_size, prediction_key):
    mamba_llm = mamba_llm.to("cuda")
    x = torch.randint(size=(batch_size, sequence_length), high=vocab_size).to("cuda")
    inputs = {"input_ids": x}
    y = mamba_llm(inputs)
    assert prediction_key in y.keys()
    assert y[prediction_key].shape == (batch_size, sequence_length, vocab_size)


def test__create_block(d_model):
    test_block = create_block(d_model=d_model)
    assert test_block.norm.normalized_shape[0] == d_model
    assert test_block.mixer.d_model == d_model


def test_tie_weights(mamba_llm):
    assert (mamba_llm.lm_head.weight != mamba_llm.backbone.embedding.weight).any()
    mamba_llm.tie_embeddings = True
    mamba_llm.tie_weights()
    assert (mamba_llm.lm_head.weight == mamba_llm.backbone.embedding.weight).all()


def test_generate(mamba_llm):
    raise NotImplementedError
