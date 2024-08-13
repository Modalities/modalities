from enum import Enum

import torch

import modalities
import modalities.util
from modalities.batch import DatasetBatch
from modalities.util import get_local_number_of_trainable_parameters, get_total_number_of_trainable_parameters


def configure_dataloader_mock(
    batch_size: int,
    seq_len: int,
    num_batches: int,
    sample_key: str,
    target_key: str,
    llm_data_loader_mock,
):
    sample_tensor = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    samples = {sample_key: sample_tensor[:, :-1]}
    targets = {target_key: sample_tensor[:, 1:]}

    batches = [DatasetBatch(targets=targets, samples=samples) for _ in range(num_batches)]

    llm_data_loader_mock.__iter__ = lambda _: iter(batches)
    llm_data_loader_mock.batch_size = batch_size
    llm_data_loader_mock.fast_forward_batch_id = 0
    llm_data_loader_mock.__len__ = lambda _: num_batches

    return llm_data_loader_mock, batches


def test_get_local_number_of_trainable_parameters():
    # Create a simple model with trainable parameters
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))

    # Calculate the expected number of trainable parameters
    expected_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Call the function and check the result
    assert get_local_number_of_trainable_parameters(model) == expected_params


def test_get_total_number_of_trainable_parameters():
    # Create a simple model with trainable parameters
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))

    # Calculate the expected number of trainable parameters
    expected_params = 10 * 5 + 5 + 5 * 2 + 2  # weights_1 + bias_1 + weights_2 + bias_2 = 67
    world_size = 8
    num_gpus_per_node = 4

    # Create a mock FSDP model
    class MockFSDP:
        class ShardingStrategy(Enum):
            FULL_SHARD = "FULL_SHARD"
            HYBRID_SHARD = "HYBRID_SHARD"

        def __init__(self, model):
            self.model = model
            self.sharding_strategy = self.ShardingStrategy.FULL_SHARD

    fsdp_model = MockFSDP(model)

    # Mock the dist.all_reduce function
    def mock_all_reduce(tensor, op):
        tensor.item = lambda: tensor
        return tensor

    def mock_cuda(tensor):
        return tensor

    def mock_world_size():
        return world_size

    def mock_device_count():
        return num_gpus_per_node

    def mock_get_local_number_of_trainable_parameters(model: MockFSDP):
        if model.sharding_strategy == MockFSDP.ShardingStrategy.FULL_SHARD:
            return get_local_number_of_trainable_parameters(model.model)
        elif model.sharding_strategy == MockFSDP.ShardingStrategy.HYBRID_SHARD:
            sharding_factor = world_size // num_gpus_per_node
            return sharding_factor * get_local_number_of_trainable_parameters(model.model)
        else:
            raise ValueError(f"Sharding strategy {model.sharding_strategy} not supported.")

    modalities.util.get_local_number_of_trainable_parameters = mock_get_local_number_of_trainable_parameters
    torch.distributed.all_reduce = mock_all_reduce
    torch.distributed.get_world_size = mock_world_size
    torch.cuda.device_count = mock_device_count
    torch.Tensor.cuda = mock_cuda

    assert get_total_number_of_trainable_parameters(fsdp_model) == expected_params

    fsdp_model.sharding_strategy = MockFSDP.ShardingStrategy.HYBRID_SHARD
    modalities.util.get_local_number_of_trainable_parameters = mock_get_local_number_of_trainable_parameters
    assert get_total_number_of_trainable_parameters(fsdp_model) == expected_params
