import torch

from modalities.models.mamba.mamba_block import MambaBlock
import pytest
from mamba_ssm.utils.generation import InferenceParams


@pytest.fixture()
def batch_size():
    return 2


@pytest.fixture()
def expansion_factor():
    return 2


@pytest.fixture()
def d_model():
    return 16


@pytest.fixture()
def d_state():
    return 3


@pytest.fixture()
def d_conv():
    return 4


@pytest.fixture()
def sequence_length():
    return 64


@pytest.fixture()
def conv_state(batch_size, expansion_factor, d_model, d_conv):
    return torch.rand((batch_size, expansion_factor * d_model, d_conv))


@pytest.fixture()
def ssm_state(batch_size, expansion_factor, d_model, d_state):
    return torch.rand((batch_size, expansion_factor * d_model, d_state))


@pytest.fixture()
def model(d_model, d_state, d_conv, expansion_factor):
    return MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expansion_factor)


@pytest.fixture()
def hidden_states(d_model,
                  batch_size, ):
    return torch.randn(batch_size, 1, d_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="We need cuda to run Mamba.")
def test_forward(batch_size,
                 sequence_length,
                 d_model,
                 d_state,
                 d_conv,
                 expansion_factor,
                 model):
    x = torch.randn(batch_size, sequence_length, d_model).to("cuda")
    model = model.to("cuda")
    y = model(x)
    assert y.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="We need cuda to run Mamba.")
def test_get_states_from_cache(conv_state,
                                  ssm_state,
                                  batch_size,
                                  expansion_factor,
                                  d_model,
                                  d_state,
                                  d_conv,
                                  model
                                  ):
    inference_params = InferenceParams(max_seqlen=16, max_batch_size=3, seqlen_offset=0, batch_size_offset=0,
                                       key_value_memory_dict={7: (conv_state, ssm_state)},
                                       lengths_per_sample=None)
    model = model.to("cuda")
    computed_conv_state, computed_ssm_state = model._get_states_from_cache(inference_params, batch_size)
    assert (conv_state == computed_conv_state).all()
    assert (ssm_state == computed_ssm_state).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="We need cuda to run Mamba.")
def test_step(conv_state, ssm_state, model, hidden_states):
    device = "cuda"
    model = model.to(device)
    hidden_states = hidden_states.to(device)
    conv_state = conv_state.to(device)
    ssm_state = ssm_state.to(device)
    computed_hidden_states, computed_conv_state, computed_ssm_state = model.step(
        hidden_states=hidden_states.detach().clone(),
        conv_state=conv_state.detach().clone(),
        ssm_state=ssm_state.detach().clone())
    assert computed_hidden_states.shape == hidden_states.shape
    assert computed_conv_state.shape == conv_state.shape
    assert computed_ssm_state.shape == ssm_state.shape
    assert (computed_hidden_states != hidden_states).any()
    assert (computed_conv_state != conv_state).any()
    assert (computed_ssm_state != ssm_state).any()


def test_allocate_inference_cache(model, batch_size, sequence_length, conv_state, ssm_state):
    device = "cuda"
    model.to(device)
    computed_conv_state, computed_ssm_state = model.allocate_inference_cache(batch_size=batch_size, max_seqlen=sequence_length,
                                                           dtype=torch.float32)
    assert (computed_conv_state == torch.zeros(conv_state.shape).to(device)).all()
    assert (computed_ssm_state == torch.zeros(ssm_state.shape).to(device)).all()


