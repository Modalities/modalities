import pytest
import torch
from transformers import AutoTokenizer
from modalities.models.mamba.mamba_model import _init_weights, create_block, MambaLLM
from tests.conftest import _ROOT_DIR


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


def test__create_block(d_model, ssm_cfg, norm_epsilon, rms_norm, residual_in_fp32, fused_add_norm, layer_idx, device,
                       dtype):
    test_block = create_block(d_model=d_model, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm,
                              residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, layer_idx=layer_idx,
                              device=device, dtype=dtype)
    assert test_block.norm.normalized_shape[0] == d_model
    assert test_block.mixer.d_model == d_model


def test_tie_weights(mamba_llm):
    assert (mamba_llm.lm_head.weight != mamba_llm.backbone.embedding.weight).any()
    mamba_llm.tie_embeddings = True
    mamba_llm.tie_weights()
    assert (mamba_llm.lm_head.weight == mamba_llm.backbone.embedding.weight).all()


def test_generate_text(d_model, n_layer, rms_norm, residual_in_fp32, fused_add_norm, prediction_key, sample_key, seed, dtype, initializer_cfg, mixer_model_config):
    mamba_llm = MambaLLM(d_model=d_model, n_layer=n_layer, vocab_size=50257, rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, pad_vocab_size_multiple=1,
                    tie_embeddings=False, prediction_key=prediction_key, sample_key=sample_key, seed=seed, dtype=dtype,
                    initializer_cfg=initializer_cfg, num_last_tokens=0, inference_params={},
                    mixer_model_config=mixer_model_config)
    tokenizer = AutoTokenizer.from_pretrained(_ROOT_DIR / "data/tokenizer/hf_gpt2")
    context = "My name is"
    output = mamba_llm.to("cuda").generate_text(tokenizer=tokenizer, context=context, max_new_tokens=5,
                                                temperature=1)
    assert type(output) == str
    assert context in output
    assert len(output) > len(context)

def test_generate(mamba_llm, vocab_size):
    num_input_tokens = 3
    max_new_tokens = 5
    input_ids = torch.randint(0, vocab_size, (1, num_input_tokens)).to("cuda")
    output = mamba_llm.to("cuda").generate(stop_token_ids=[],input_ids=input_ids, max_new_tokens=max_new_tokens,temperature=1)

    assert type(output) == torch.Tensor
    assert output.shape[1] == num_input_tokens + max_new_tokens

