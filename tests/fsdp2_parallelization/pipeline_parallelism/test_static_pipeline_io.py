import pytest
import torch

from modalities.models.parallelism.pipeline_parallelism import PipelineFactory, StaticStageIOSpec
from modalities.models.parallelism.pipeline_parallelism_configs import (
    STATIC_IO_FIELD_NAMES,
    validate_static_io_settings,
)
from modalities.running_env.env_utils import PyTorchDtypes

COMPLETE_STATIC_IO_SETTINGS = {
    "static_io_microbatch_size": 2,
    "static_io_sequence_length": 8,
    "static_io_hidden_dim": 16,
    "static_io_vocab_size": 32,
    "static_io_dtype": PyTorchDtypes.BF_16,
}


def test_complete_static_io_settings_are_valid():
    validate_static_io_settings(static_io_settings=COMPLETE_STATIC_IO_SETTINGS, expert_parallel_degree=2)


def test_absent_static_io_settings_are_valid_without_expert_parallelism():
    validate_static_io_settings(
        static_io_settings={name: None for name in STATIC_IO_FIELD_NAMES}, expert_parallel_degree=1
    )


@pytest.mark.parametrize("omitted_field", STATIC_IO_FIELD_NAMES)
def test_partial_static_io_settings_are_rejected(omitted_field: str):
    """A single omitted field must not silently fall back to dynamic shape inference, which
    deadlocks under expert parallelism."""
    static_io_settings = {**COMPLETE_STATIC_IO_SETTINGS, omitted_field: None}

    with pytest.raises(ValueError, match="completely or not at all"):
        validate_static_io_settings(static_io_settings=static_io_settings, expert_parallel_degree=1)


def test_expert_parallelism_without_static_io_settings_is_rejected():
    with pytest.raises(ValueError, match="Expert parallelism combined with pipeline parallelism"):
        validate_static_io_settings(
            static_io_settings={name: None for name in STATIC_IO_FIELD_NAMES}, expert_parallel_degree=2
        )


@pytest.mark.parametrize("dtype", [PyTorchDtypes.BF_16, PyTorchDtypes.FP_16, PyTorchDtypes.FP_32])
def test_static_stage_io_uses_the_configured_activation_dtype(dtype: PyTorchDtypes):
    """The example activations must carry the configured dtype (not a hardcoded one), since the
    pipeline allocates its P2P buffers from this metadata."""
    static_io_spec = StaticStageIOSpec(
        microbatch_size=2, sequence_length=8, hidden_dim=16, vocab_size=32, activation_dtype=dtype.value
    )

    first_input, first_output = PipelineFactory._build_static_stage_io(
        stage_idx=0, num_stages=3, static_io_spec=static_io_spec, device=torch.device("cpu")
    )
    middle_input, middle_output = PipelineFactory._build_static_stage_io(
        stage_idx=1, num_stages=3, static_io_spec=static_io_spec, device=torch.device("cpu")
    )
    last_input, last_output = PipelineFactory._build_static_stage_io(
        stage_idx=2, num_stages=3, static_io_spec=static_io_spec, device=torch.device("cpu")
    )

    # The first stage consumes token ids, which are integer and carry no gradient.
    assert first_input.shape == (2, 8)
    assert first_input.dtype == torch.long
    assert not first_input.requires_grad

    # All hidden states use the configured activation dtype and require gradients, so that the
    # pipeline allocates the backward (gradient) P2P buffers as well.
    for hidden_state in (first_output, middle_input, middle_output, last_input):
        assert hidden_state.shape == (2, 8, 16)
        assert hidden_state.dtype == dtype.value
        assert hidden_state.requires_grad

    # The last stage emits logits.
    assert last_output.shape == (2, 8, 32)
    assert last_output.dtype == dtype.value
    assert last_output.requires_grad
