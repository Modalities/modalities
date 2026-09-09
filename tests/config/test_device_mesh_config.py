"""Validation tests for :class:`DeviceMeshConfig`, focused on the expert parallel degree.

Expert parallelism is carved out of the data-parallel shard dimension rather than multiplying into
the world size, which makes its constraints easy to get wrong in a config. These tests pin them down
without needing a process group.
"""

import pytest

from modalities.exceptions import ConfigError
from modalities.running_env.fsdp.device_mesh import DeviceMeshConfig


def _config(**overrides) -> DeviceMeshConfig:
    kwargs = dict(data_parallel_shard_degree=-1, world_size=8)
    return DeviceMeshConfig(**{**kwargs, **overrides})


def test_expert_parallel_degree_defaults_to_one():
    assert _config().expert_parallel_degree == 1


def test_expert_parallelism_does_not_consume_world_size():
    # dp_shard resolves to the full world size even though 4 ranks' worth of it carries the experts.
    config = _config(data_parallel_shard_degree=-1, expert_parallel_degree=4)
    assert config.data_parallel_shard_degree == 8
    assert config.expert_parallel_degree == 4


@pytest.mark.parametrize("expert_parallel_degree", [1, 2, 4, 8])
def test_expert_parallel_degree_dividing_dp_shard_is_accepted(expert_parallel_degree: int):
    _config(data_parallel_shard_degree=8, expert_parallel_degree=expert_parallel_degree)


def test_expert_parallel_degree_exceeding_dp_shard_is_rejected():
    with pytest.raises(ConfigError, match="must not exceed data_parallel_shard_degree"):
        _config(data_parallel_shard_degree=2, world_size=2, expert_parallel_degree=4)


def test_expert_parallel_degree_not_dividing_dp_shard_is_rejected():
    with pytest.raises(ConfigError, match="must be divisible by"):
        _config(data_parallel_shard_degree=8, expert_parallel_degree=3)


@pytest.mark.parametrize(
    "degree_name", ["tensor_parallel_degree", "pipeline_parallel_degree", "context_parallel_degree"]
)
def test_expert_parallelism_rejects_unsupported_combinations(degree_name: str):
    with pytest.raises(ConfigError, match="not yet supported together with"):
        _config(data_parallel_shard_degree=4, expert_parallel_degree=2, **{degree_name: 2})
