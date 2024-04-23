import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType, PydanticLLMDataLoaderIFType, load_app_config_dict
from modalities.running_env.cuda_env import CudaEnv
from tests.dataloader.dummy_sequential_dataset import TestDataset, TestDatasetConfig


class DataloaderInstantiationModel(BaseModel):
    train_dataloader: PydanticLLMDataLoaderIFType


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_resumable_dataloader_without_shuffling():
    # we test that the distributed sampler provides each process with the correct subset of the dataset
    # Given a sequence of [0, 1, 2, 3, 4, 5, 6, 7, 8] we want each of the two processes
    # to receive [[0, 2], [4, 6]] and [[1, 3], [5, 7]], respectively.

    config_file_path = Path("tests/dataloader/distributed/dist_dataloader_config_without_shuffling.yaml")
    config_dict = load_app_config_dict(config_file_path)

    main = Main(config_dict, config_file_path)
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main.add_custom_component(
            component_key="dataset",
            variant_key="test",
            custom_component=TestDataset,
            custom_config=TestDatasetConfig,
        )
        components = main.build_components(components_model_type=DataloaderInstantiationModel)

        train_dataloader = components.train_dataloader
        num_samples = len(train_dataloader.dataset)

        batches = [batch.tolist() for batch in train_dataloader]

        rank = dist.get_rank()
        with open(f"tests/tmp/rank_{rank}_batches.json", "w") as f:
            json.dump(batches, f)

        dist.barrier()

        with open("tests/tmp/rank_0_batches.json") as f:
            rank_0_batches = torch.tensor(json.load(f))

        with open("tests/tmp/rank_1_batches.json") as f:
            rank_1_batches = torch.tensor(json.load(f))

        samples = [i.item() for item in zip(rank_0_batches.flatten(), rank_1_batches.flatten()) for i in item]

        assert len(rank_1_batches.flatten()) == num_samples // 2
        assert len(samples) == num_samples and num_samples > 0
        assert samples == list(range(num_samples))


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_resumable_dataloader_with_shuffling():
    # we test that the distributed sampler provides each process with the correct RANDOM subset of the dataset
    # Given a sequence of [0, 1, 2, 3, 4, 5, 6, 7, 8] we want each of the two processes
    # to receive two batches of size two without overlap, e.g., [[2, 0], [5, 6]] and [[7, 3], [4, 1]], respectively.

    config_file_path = Path("tests/dataloader/distributed/dist_dataloader_config_with_shuffling.yaml")
    config_dict = load_app_config_dict(config_file_path)

    main = Main(config_dict, config_file_path)
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main.add_custom_component(
            component_key="dataset",
            variant_key="test",
            custom_component=TestDataset,
            custom_config=TestDatasetConfig,
        )
        components = main.build_components(components_model_type=DataloaderInstantiationModel)

        train_dataloader = components.train_dataloader
        num_samples = len(train_dataloader.dataset)

        batches = [batch.tolist() for batch in train_dataloader]

        rank = dist.get_rank()
        with open(f"tests/tmp/rank_{rank}_batches.json", "w") as f:
            json.dump(batches, f)

        dist.barrier()

        with open("tests/tmp/rank_0_batches.json") as f:
            rank_0_batches = torch.tensor(json.load(f))

        with open("tests/tmp/rank_1_batches.json") as f:
            rank_1_batches = torch.tensor(json.load(f))

        samples = [i.item() for item in zip(rank_0_batches.flatten(), rank_1_batches.flatten()) for i in item]

        assert len(rank_1_batches.flatten()) == num_samples // 2
        assert len(set(samples)) == num_samples and num_samples > 0
        assert set(samples).intersection(set(range(num_samples))) == set(range(num_samples))


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_resumable_dataloader_with_shuffling_and_skipped_batches():
    # we test that the distributed sampler provides each process with the correct RANDOM subset of the dataset
    # additionally we skip one batch
    # Given a sequence of [0, 1, 2, 3, 4, 5, 6, 7, 8] we want each of the two processes
    # to receive one batch of size two without overlap, e.g., [[5, 6]] and [[4, 1]], respectively.

    config_shuffled_file_path = Path("tests/dataloader/distributed/dist_dataloader_config_with_shuffling.yaml")
    config_shuffled_dict = load_app_config_dict(config_shuffled_file_path)
    main_shuffled = Main(config_shuffled_dict, config_shuffled_file_path)

    config_shuffled_and_skipped_file_path = Path(
        "tests/dataloader/distributed/dist_dataloader_config_with_shuffling_and_skipped_batches.yaml"
    )
    config_shuffled_and_skipped_dict = load_app_config_dict(config_shuffled_and_skipped_file_path)
    main_shuffled_and_skipped = Main(config_shuffled_and_skipped_dict, config_shuffled_and_skipped_file_path)

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main_shuffled.add_custom_component(
            component_key="dataset",
            variant_key="test",
            custom_component=TestDataset,
            custom_config=TestDatasetConfig,
        )
        components_shuffled = main_shuffled.build_components(components_model_type=DataloaderInstantiationModel)
        train_dataloader_shuffled = components_shuffled.train_dataloader
        batches_shuffled = [batch.tolist() for batch in train_dataloader_shuffled]

        main_shuffled_and_skipped.add_custom_component(
            component_key="dataset",
            variant_key="test",
            custom_component=TestDataset,
            custom_config=TestDatasetConfig,
        )
        components_shuffled_and_skipped = main_shuffled_and_skipped.build_components(
            components_model_type=DataloaderInstantiationModel
        )
        train_dataloader_shuffled_and_skipped = components_shuffled_and_skipped.train_dataloader
        batches_shuffled_and_skipped = [batch.tolist() for batch in train_dataloader_shuffled_and_skipped]

        rank = dist.get_rank()
        with open(f"tests/tmp/rank_{rank}_batches_shuffled.json", "w") as f:
            json.dump(batches_shuffled, f)
        with open(f"tests/tmp/rank_{rank}_batches_shuffled_and_skipped.json", "w") as f:
            json.dump(batches_shuffled_and_skipped, f)

        dist.barrier()

        with open("tests/tmp/rank_0_batches_shuffled.json") as f:
            rank_0_batches_shuffled = torch.tensor(json.load(f))

        with open("tests/tmp/rank_1_batches_shuffled.json") as f:
            rank_1_batches_shuffled = torch.tensor(json.load(f))

        with open("tests/tmp/rank_0_batches_shuffled_and_skipped.json") as f:
            rank_0_batches_shuffled_and_skipped = torch.tensor(json.load(f))

        with open("tests/tmp/rank_1_batches_shuffled_and_skipped.json") as f:
            rank_1_batches_shuffled_and_skipped = torch.tensor(json.load(f))

        assert all(rank_0_batches_shuffled[-1] == rank_0_batches_shuffled_and_skipped[0])
        assert all(rank_1_batches_shuffled[-1] == rank_1_batches_shuffled_and_skipped[0])
