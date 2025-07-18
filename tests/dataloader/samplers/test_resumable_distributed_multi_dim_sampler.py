import pytest

from modalities.dataloader.sampler_factory import SamplerFactory
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees


class MockMeshEntry:
    def __init__(self, coordinate: int, size):
        self._coordinate = coordinate
        self._size = size

    def get_coordinate(self) -> tuple[int]:
        return [self._coordinate]

    def size(self):
        return self._size


class MockDeviceMesh(dict):
    def __init__(self, mesh_dict: dict[ParallelismDegrees, tuple[int, int]]):
        super().__init__()
        for key, (coord, size) in mesh_dict.items():
            self[key] = MockMeshEntry(coord, size)


@pytest.mark.parametrize(
    "device_mesh, expected_samples",
    [
        # Test setup:
        # tp degree: 4, dp degree: 2
        # [[0, 1, 2, 3]    DP 1
        #  [4, 5, 6, 7]]   DP 2
        # DP 1 and DP2 must see different data and within each DP group the data is equivalent across all ranks
        (
            MockDeviceMesh({ParallelismDegrees.TP.value: (0, 4), ParallelismDegrees.DP_SHARD.value: (0, 2)}),
            list(range(0, 32, 2)),
        ),
        (
            MockDeviceMesh({ParallelismDegrees.TP.value: (1, 4), ParallelismDegrees.DP_SHARD.value: (0, 2)}),
            list(range(0, 32, 2)),
        ),
        (
            MockDeviceMesh({ParallelismDegrees.TP.value: (2, 4), ParallelismDegrees.DP_SHARD.value: (0, 2)}),
            list(range(0, 32, 2)),
        ),
        (
            MockDeviceMesh({ParallelismDegrees.TP.value: (3, 4), ParallelismDegrees.DP_SHARD.value: (0, 2)}),
            list(range(0, 32, 2)),
        ),
        (
            MockDeviceMesh({ParallelismDegrees.TP.value: (0, 4), ParallelismDegrees.DP_SHARD.value: (1, 2)}),
            list(range(1, 32, 2)),
        ),
        (
            MockDeviceMesh({ParallelismDegrees.TP.value: (1, 4), ParallelismDegrees.DP_SHARD.value: (1, 2)}),
            list(range(1, 32, 2)),
        ),
        (
            MockDeviceMesh({ParallelismDegrees.TP.value: (2, 4), ParallelismDegrees.DP_SHARD.value: (1, 2)}),
            list(range(1, 32, 2)),
        ),
        (
            MockDeviceMesh({ParallelismDegrees.TP.value: (3, 4), ParallelismDegrees.DP_SHARD.value: (1, 2)}),
            list(range(1, 32, 2)),
        ),
    ],
)
def test_resumable_distributed_multi_dim_sampler(device_mesh: MockDeviceMesh, expected_samples: list[int]):
    dataset = list(range(32))

    sampler = SamplerFactory.create_resumable_distributed_multi_dim_sampler(
        dataset=dataset,
        device_mesh=device_mesh,
        data_parallel_key=ParallelismDegrees.DP_SHARD,
        epoch=0,
        shuffle=False,
        seed=0,
        drop_last=True,
        skip_num_global_samples=0,
    )

    actual_samples = list(sampler)
    assert actual_samples == expected_samples, f"Expected {expected_samples}, but got {actual_samples}"
