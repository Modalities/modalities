import torch
import pytest
from mamba_ssm.utils.generation import InferenceParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="We need cuda to run Mamba.")
def test_mamba_block_forward(batch_size,
                             sequence_length,
                             d_model,
                             d_state,
                             d_conv,
                             expand,
                             mamba_block):
    x = torch.randn(batch_size, sequence_length, d_model).to("cuda")
    mamba_block = mamba_block.to("cuda")
    y = mamba_block(x)
    assert y.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="We need cuda to run Mamba.")
def test_block_forward(hidden_states, block):
    block = block.to("cuda")
    hidden_states = hidden_states.to("cuda")
    computed_hidden_states, computed_residuals = block(hidden_states)
    assert (hidden_states == computed_residuals).all()
    assert hidden_states.shape == computed_hidden_states.shape
    assert (hidden_states != computed_hidden_states).any()


def test_get_states_from_cache(conv_state,
                               ssm_state,
                               batch_size,
                               expand,
                               d_model,
                               d_state,
                               d_conv,
                               mamba_block,
                               layer_idx
                               ):
    inference_params = InferenceParams(max_seqlen=16, max_batch_size=3, seqlen_offset=0, batch_size_offset=0,
                                       key_value_memory_dict={layer_idx: (conv_state, ssm_state)},
                                       lengths_per_sample=None)
    computed_conv_state, computed_ssm_state = mamba_block._get_states_from_cache(inference_params, batch_size)
    assert (conv_state == computed_conv_state).all()
    assert (ssm_state == computed_ssm_state).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="We need cuda to run Mamba.")
def test_step(conv_state, ssm_state, mamba_block, hidden_states):
    device = "cuda"
    mamba_block = mamba_block.to(device)
    hidden_states = hidden_states.to(device)
    conv_state = conv_state.to(device)
    ssm_state = ssm_state.to(device)
    computed_hidden_states, computed_conv_state, computed_ssm_state = mamba_block.step(
        hidden_states=hidden_states.detach().clone(),
        conv_state=conv_state.detach().clone(),
        ssm_state=ssm_state.detach().clone())
    assert computed_hidden_states.shape == hidden_states.shape
    assert computed_conv_state.shape == conv_state.shape
    assert computed_ssm_state.shape == ssm_state.shape
    assert (computed_hidden_states != hidden_states).any()
    assert (computed_conv_state != conv_state).any()
    assert (computed_ssm_state != ssm_state).any()


def test_allocate_inference_cache(mamba_block, batch_size, sequence_length, conv_state, ssm_state):
    device = "cuda"
    mamba_block.to(device)
    computed_conv_state, computed_ssm_state = mamba_block.allocate_inference_cache(batch_size=batch_size,
                                                                                   max_seqlen=sequence_length,
                                                                                   dtype=torch.float32)
    assert (computed_conv_state == torch.zeros(conv_state.shape).to(device)).all()
    assert (computed_ssm_state == torch.zeros(ssm_state.shape).to(device)).all()
