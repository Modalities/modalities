import os
from unittest.mock import MagicMock

import pytest
from torch.utils.data import BatchSampler

from modalities.dataloader.dataloader_factory import DataloaderFactory
from modalities.dataloader.sampler_factory import SamplerFactory
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from tests.dataloader.distributed.mocks import MultiProcessingCudaEnvMock
from tests.dataloader.dummy_sequential_dataset import TestDataset


@pytest.mark.parametrize("world_size, dp_degree", [(4, 2)])
def test_distributed_multidim_dataloader_produces_same_data_on_connected_non_dp_ranks(world_size: int, dp_degree: int):
    batches_on_rank = _build_batch_for_each_rank_combination(world_size, dp_degree)

    for dp_rank in range(dp_degree):
        assert all(
            batches_on_rank[(dp_rank, 0)] == batches_on_rank[(dp_rank, other_rank)]
            for other_rank in range(1, world_size // dp_degree)
        ), f"Batches on dp_rank {dp_rank} differ across other ranks."


@pytest.mark.parametrize("world_size, dp_degree", [(4, 2)])
def test_distributed_multidim_dataloader_produces_different_data_on_different_dp_ranks(world_size: int, dp_degree: int):
    batches_on_rank = _build_batch_for_each_rank_combination(world_size, dp_degree)

    for dp_rank1 in range(dp_degree):
        for dp_rank2 in range(dp_rank1 + 1, dp_degree):
            samples_dp_rank1 = sum(batches_on_rank[(dp_rank1, 0)], [])
            samples_dp_rank2 = sum(batches_on_rank[(dp_rank2, 0)], [])
            assert (
                len(set(samples_dp_rank1).intersection(samples_dp_rank2)) == 0
            ), f"Data samples on different data parallel ranks {dp_rank1} and {dp_rank2} should be disjoint."


@pytest.mark.parametrize("world_size, dp_degree", [(4, 2)])
def test_distributed_multidim_dataloader_produces_expected_samples(world_size: int, dp_degree: int):
    dataset_len = 16
    batches_on_rank = _build_batch_for_each_rank_combination(world_size, dp_degree, dataset_len)

    for dp_rank in range(dp_degree):
        samples_dp_rank = sum(batches_on_rank[(dp_rank, 0)], [])
        expected_samples_on_dp_rank = list(range(dp_rank, dataset_len, dp_degree))

        assert set(samples_dp_rank) == set(
            expected_samples_on_dp_rank
        ), f"Data samples on dp_rank {dp_rank} do not match expected samples."


def _build_batch_for_each_rank_combination(
    world_size: int, dp_degree: int, dataset_len: int = 16
) -> dict[tuple[int, int], list[list[int]]]:
    return {
        (dp_rank, other_rank): _load_data_for_ranks(dp_rank, other_rank, world_size, dp_degree, dataset_len)
        for dp_rank, other_rank in _get_rank_combinations(world_size, dp_degree)
    }


def _get_other_degree(world_size: int, dp_degree: int) -> int:
    return world_size // dp_degree


def _get_rank_combinations(world_size: int, dp_degree: int) -> list[tuple[int, int]]:
    other_degree = _get_other_degree(world_size, dp_degree)
    return [(dp_rank, other_rank) for dp_rank in range(dp_degree) for other_rank in range(other_degree)]


def _load_data_for_ranks(
    dp_rank: int, other_rank: int, world_size: int, dp_degree: int, dataset_len: int
) -> list[list[int]]:
    global_rank = dp_rank * _get_other_degree(world_size, dp_degree) + other_rank
    with MultiProcessingCudaEnvMock(
        global_rank=global_rank,
        local_rank=other_rank,
        world_size=world_size,
        rdvz_port=22350,
    ):
        device_mesh = _build_device_mesh_mock(world_size, dp_degree, dp_rank, other_rank)
        dataset = TestDataset(dataset_len)
        sampler = SamplerFactory.create_resumable_distributed_multi_dim_sampler(
            dataset=dataset, device_mesh=device_mesh, data_parallel_key=ParallelismDegrees.DP_SHARD
        )
        batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=True)
        train_dataloader = DataloaderFactory.get_dataloader(
            dataloader_tag="train",
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=None,
            num_workers=2,
            pin_memory=False,
        )
        return [batch.tolist() for batch in train_dataloader]


def _build_device_mesh_mock(world_size: int, dp_degree: int, dp_rank: int, other_rank: int) -> dict[str, MagicMock]:
    dp_device_mesh = MagicMock()
    dp_device_mesh.size.return_value = dp_degree
    dp_device_mesh.get_coordinate.return_value = [dp_rank]
    other_device_mesh = MagicMock()
    other_degree = world_size // dp_degree
    other_device_mesh.size.return_value = int(os.environ["WORLD_SIZE"]) // other_degree
    other_device_mesh.get_coordinate.return_value = [other_rank]
    device_mesh_mock = {ParallelismDegrees.DP_SHARD.value: dp_device_mesh, "other": other_device_mesh}
    return device_mesh_mock
