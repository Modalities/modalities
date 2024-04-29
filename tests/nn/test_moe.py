import pytest
import torch
import torch.nn as nn

from modalities.nn.moe import MoEExpertGLU, MoEExperts, MoEFFN, MoEFFNConfig, MoERouter


def test_moe_router_produces_expected_shapes(
    model_input: torch.Tensor,
    batch_size: int,
    seq_length: int,
    moe_top_k: int,
    moe_router: MoERouter,
):
    top_weights, top_experts = moe_router.forward(model_input)
    assert top_weights.shape == (batch_size * seq_length, moe_top_k)
    assert top_experts.shape == (batch_size * seq_length, moe_top_k)


def test_moe_router_jitter_does_not_change_shape(model_input: torch.Tensor, moe_router: MoERouter):
    jittered_model_input = moe_router._jitter(model_input)
    assert jittered_model_input.shape == model_input.shape


def test_moe_expert_produces_expected_shape(
    model_input: torch.Tensor, batch_size: int, seq_length: int, hidden_size: int, moe_num_experts: int
):
    ffn_hidden_size = 128
    act_fn = nn.ReLU
    model = MoEExpertGLU(hidden_size, ffn_hidden_size, moe_num_experts, act_fn)
    expert_idx = 2
    output = model.forward(model_input, expert_idx)
    assert output.shape == (batch_size, seq_length, hidden_size)


def test_moe_expert_errors_with_invalid_idx(model_input: torch.Tensor, hidden_size: int, moe_num_experts: int):
    ffn_hidden_size = 128
    act_fn = nn.ReLU

    model = MoEExpertGLU(hidden_size, ffn_hidden_size, moe_num_experts, act_fn)
    invalid_expert_idx = moe_num_experts + 2

    with pytest.raises(IndexError):
        model.forward(model_input, invalid_expert_idx)


def test_moe_experts_produce_expected_shape(
    batch_size: int, seq_length: int, hidden_size: int, moe_num_experts: int, moe_top_k: int
):
    ffn_hidden_size = 64
    act_fn = nn.ReLU

    model = MoEExperts(hidden_size, ffn_hidden_size, moe_num_experts, act_fn)
    x = torch.rand(batch_size, seq_length, hidden_size)
    top_weights = torch.rand(batch_size * seq_length, moe_top_k)
    top_experts = torch.randint(0, moe_top_k, (batch_size * seq_length, moe_top_k))

    output = model(x, top_weights, top_experts)

    assert output.shape == (batch_size, seq_length, hidden_size)


def test_moeffn_output_shape(batch_size: int, seq_length: int, moe_config: MoEFFNConfig):
    hidden_router_size = 128
    model = MoEFFN(hidden_router_size, moe_config)
    input_tensor = torch.randn(batch_size, seq_length, hidden_router_size)
    output = model(input_tensor)
    assert output.shape == (batch_size, seq_length, hidden_router_size)


@pytest.fixture
def moe_router(hidden_size: int, moe_config: MoEFFNConfig) -> MoERouter:
    return MoERouter(hidden_size, moe_config)


@pytest.fixture
def moe_config(moe_num_experts: int, moe_top_k: int, uniform_expert_assignment: bool) -> MoEFFNConfig:
    return MoEFFNConfig(
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        moe_normalize_expert_weights=2.0,
        uniform_expert_assignment=uniform_expert_assignment,
        ffn_hidden_size=128,
        act_fn=nn.ReLU,
        moe_jitter_eps=0.1,
    )


@pytest.fixture
def model_input(batch_size: int, seq_length: int, hidden_size: int) -> torch.Tensor:
    return torch.randn(batch_size, seq_length, hidden_size)


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def seq_length() -> int:
    return 32


@pytest.fixture
def hidden_size() -> int:
    return 10


@pytest.fixture
def moe_num_experts() -> int:
    return 5


@pytest.fixture
def moe_top_k() -> int:
    return 3


@pytest.fixture(params=[True, False])
def uniform_expert_assignment(request: pytest.FixtureRequest) -> bool:
    return request.param
