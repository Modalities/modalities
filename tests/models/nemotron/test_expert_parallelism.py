"""Unit tests for the expert-parallel dispatch permutation.

The permutation that regroups received tokens from sender-major to expert-major order is the one
piece of expert parallelism that is pure index arithmetic, so it is tested here without a process
group. The distributed behaviour is covered by
``tests/fsdp2_parallelization/test_expert_parallelism_fsdp2.py``.
"""

import pytest
import torch

from modalities.models.parallelism.expert_parallelism import _build_permute_indices


def _reference_permute_indices(counts: torch.Tensor) -> torch.Tensor:
    """Straightforward loop reference: concatenate the (expert, sender) ranges in expert-major order.

    Args:
        counts (torch.Tensor): Received token counts of shape ``(ep_degree, num_local_experts)``.

    Returns:
        torch.Tensor: The expected gather indices.
    """
    ep_degree, num_local_experts = counts.shape
    flat = counts.reshape(-1)
    starts = (flat.cumsum(0) - flat).view(ep_degree, num_local_experts)
    ranges = [
        torch.arange(int(starts[sender, expert]), int(starts[sender, expert]) + int(counts[sender, expert]))
        for expert in range(num_local_experts)
        for sender in range(ep_degree)
    ]
    return torch.cat(ranges) if ranges else torch.zeros(0, dtype=torch.long)


def _build(counts: torch.Tensor) -> torch.Tensor:
    ep_degree, num_local_experts = counts.shape
    return _build_permute_indices(
        tokens_per_expert_group=counts.reshape(-1),
        ep_degree=ep_degree,
        num_local_experts=num_local_experts,
        total_tokens=int(counts.sum()),
    )


@pytest.mark.parametrize(
    "counts",
    [
        pytest.param(torch.tensor([[3, 1], [2, 4]]), id="balanced_2x2"),
        pytest.param(torch.tensor([[0, 5], [7, 0]]), id="some_experts_empty"),
        pytest.param(torch.tensor([[0, 0], [0, 0]]), id="all_empty"),
        pytest.param(torch.tensor([[1]]), id="degenerate_single_block"),
        pytest.param(torch.tensor([[2, 0, 3, 1], [0, 0, 0, 0], [1, 1, 1, 1]]), id="one_sender_silent"),
    ],
)
def test_permute_indices_match_loop_reference(counts: torch.Tensor):
    torch.testing.assert_close(_build(counts), _reference_permute_indices(counts))


def test_permute_indices_are_a_permutation():
    torch.manual_seed(0)
    counts = torch.randint(0, 9, (4, 6))
    indices = _build(counts)
    total = int(counts.sum())
    assert indices.shape == (total,)
    # A valid gather permutation visits every row of the receive buffer exactly once.
    torch.testing.assert_close(indices.sort().values, torch.arange(total))


def test_permute_indices_group_tokens_by_expert():
    """The permuted order must place all tokens of one expert in one contiguous block."""
    counts = torch.tensor([[2, 1], [3, 4]])  # 2 senders, 2 local experts
    indices = _build(counts)
    # Sender-major receive buffer labelled by the expert each row belongs to.
    expert_of_row = torch.tensor([0, 0, 1, 0, 0, 0, 1, 1, 1, 1])
    permuted = expert_of_row[indices]
    torch.testing.assert_close(permuted, torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))


def test_permute_indices_do_not_read_values_back_to_host(monkeypatch):
    """The construction must not read any tensor value back to the host.

    A device-to-host copy per (sender, local expert) pair is what makes the obvious implementation
    expensive: it costs thousands of synchronizations per step for a deep MoE model. Only
    ``total_tokens`` may come from the host, and the caller passes it in because it already has it
    from the all-to-all split lists.
    """
    counts = torch.randint(0, 9, (4, 8))
    total = int(counts.sum())

    observed: list[str] = []
    for name in ("item", "tolist", "numpy"):
        original = getattr(torch.Tensor, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            observed.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, spy)

    _build_permute_indices(
        tokens_per_expert_group=counts.reshape(-1),
        ep_degree=4,
        num_local_experts=8,
        total_tokens=total,
    )
    assert observed == [], f"permutation construction read values back to the host via {set(observed)}"
