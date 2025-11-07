import pytest
import torch
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import BatchSampler

from modalities.config.config import ProcessGroupBackendType
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.dataloader_factory import DataloaderFactory
from modalities.dataloader.sampler_factory import SamplerFactory
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_device_mesh, get_mesh_for_parallelism_method
from tests.dataloader.distributed.mocks import MultiProcessingCudaEnvMock
from tests.dataloader.dummy_sequential_dataset import TestDataset
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv
from tests.mocks import MockDeviceMesh
from tests.utility import find_free_port, tensors_equal_across_mesh, tensors_pairwise_not_equal_across_mesh


@pytest.mark.parametrize("world_size, dp_degree", [(4, 2)])
def test_dataloader_produces_same_data_on_connected_non_dp_ranks(world_size: int, dp_degree: int):
    batches_on_rank = _build_batch_for_each_rank_combination(world_size, dp_degree)

    for dp_rank in range(dp_degree):
        assert all(
            batches_on_rank[(dp_rank, 0)] == batches_on_rank[(dp_rank, other_rank)]
            for other_rank in range(1, world_size // dp_degree)
        ), f"Batches on dp_rank {dp_rank} differ across other ranks."


@pytest.mark.parametrize("world_size, dp_degree", [(4, 2)])
def test_dataloader_produces_different_data_on_different_dp_ranks(world_size: int, dp_degree: int):
    batches_on_rank = _build_batch_for_each_rank_combination(world_size, dp_degree)

    for dp_rank1 in range(dp_degree):
        for dp_rank2 in range(dp_rank1 + 1, dp_degree):
            samples_dp_rank1 = sum(batches_on_rank[(dp_rank1, 0)], [])
            samples_dp_rank2 = sum(batches_on_rank[(dp_rank2, 0)], [])
            assert (
                len(set(samples_dp_rank1).intersection(samples_dp_rank2)) == 0
            ), f"Data samples on different data parallel ranks {dp_rank1} and {dp_rank2} should be disjoint."


@pytest.mark.parametrize("world_size, dp_degree", [(4, 2)])
def test_dataloader_produces_expected_samples(world_size: int, dp_degree: int):
    dataset_len = 16
    batches_on_rank = _build_batch_for_each_rank_combination(world_size, dp_degree, dataset_len)

    for dp_rank in range(dp_degree):
        batch_sizes = set(len(batch) for batch in batches_on_rank[(dp_rank, 0)])
        assert len(batch_sizes) == 1, f"Batches on dp_rank {dp_rank} have different sizes."
        samples_dp_rank = sum(batches_on_rank[(dp_rank, 0)], [])
        expected_samples_on_dp_rank = list(range(dp_rank, dataset_len, dp_degree))

        assert set(samples_dp_rank) == set(
            expected_samples_on_dp_rank
        ), f"Data samples on dp_rank {dp_rank} do not match expected samples."


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="This test requires at least 8 GPUs.")
def test_dataloader_produces_different_samples_in_different_dp_ranks():
    world_size = 8
    mp.spawn(
        _test_dataloader_produces_different_samples_in_different_dp_ranks,
        args=(world_size, find_free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="This test requires at least 8 GPUs.")
def test_dataloader_produces_same_samples_in_connected_non_dp_ranks():
    world_size = 8
    mp.spawn(
        _test_dataloader_produces_same_samples_in_connected_non_dp_ranks,
        args=(world_size, find_free_port()),
        nprocs=world_size,
        join=True,
    )


def _test_dataloader_produces_different_samples_in_different_dp_ranks(process_id: int, world_size: int, port: int):
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=world_size,
        rdvz_port=port,
    ):
        device_mesh = get_device_mesh(
            device_type="cuda",
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=2,
            context_parallel_degree=1,
            enable_loss_parallel=False,
            world_size=world_size,
        )
        dataloader = _build_dataloader_for_mesh(32, device_mesh)
        dp_mesh = get_mesh_for_parallelism_method(device_mesh, ParallelismDegrees.DP_SHARD)
        for batch in dataloader:
            assert tensors_pairwise_not_equal_across_mesh(batch.to("cuda"), dp_mesh)


def _test_dataloader_produces_same_samples_in_connected_non_dp_ranks(process_id: int, world_size: int, port: int):
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=world_size,
        rdvz_port=port,
    ):
        device_mesh = get_device_mesh(
            device_type="cuda",
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=2,
            context_parallel_degree=1,
            enable_loss_parallel=False,
            world_size=world_size,
        )
        dataloader = _build_dataloader_for_mesh(32, device_mesh)
        pp_mesh = get_mesh_for_parallelism_method(device_mesh, ParallelismDegrees.PP)
        for batch in dataloader:
            assert tensors_equal_across_mesh(batch.to("cuda"), pp_mesh)
        tp_mesh = get_mesh_for_parallelism_method(device_mesh, ParallelismDegrees.TP)
        for batch in dataloader:
            assert tensors_equal_across_mesh(batch.to("cuda"), tp_mesh)


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
        device_mesh_mock = _build_device_mesh_mock(world_size, dp_degree, dp_rank, other_rank)
        train_dataloader = _build_dataloader_for_mesh(dataset_len, device_mesh_mock)
        return [batch.tolist() for batch in train_dataloader]


def _build_dataloader_for_mesh(dataset_len: int, device_mesh: DeviceMesh) -> LLMDataLoader:
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
    return train_dataloader


def _build_device_mesh_mock(world_size: int, dp_degree: int, dp_rank: int, other_rank: int) -> MockDeviceMesh:
    other_degree = world_size // dp_degree
    device_mesh_setup = {ParallelismDegrees.DP_SHARD.value: (dp_rank, dp_degree), "other": (other_rank, other_degree)}
    return MockDeviceMesh(device_mesh_setup)
