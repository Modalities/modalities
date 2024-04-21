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
    # we test that the distributed sampler provides each process with the correct subset of the dataset.
    # In the first epoch we expect the first step to be skipped but for the subsequent epochs we expect
    # all dataset samples.
    # Given a sequence of [0, 1, 2, 3, 4, 5, 6, 7, 8] we want each of the two processes to have the
    # following batches after three epochs
    # to receive [[4, 6], [0, 2], [4, 6], [0, 2], [4, 6]] and
    #  [[5, 7], [1, 3], [5, 7], [1, 3], [5, 7]], respectively.

    config_file_path = Path(
        "tests/dataloader/distributed/dist_repeating_dataloader_config_without_shuffling_but_skipped_batch.yaml"
    )
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

        repeating_dataloader = components.train_dataloader
        num_samples = len(repeating_dataloader.dataloader.dataset)

        # each epoch has 2 batches of size 2, we want two skip the first batch in the
        # first epoch and have 3 epochs in total
        num_batches = 5

        batches = [batch.tolist() for _, batch in zip(range(num_batches), repeating_dataloader)]

        rank = dist.get_rank()
        with open(f"tests/tmp/rank_{rank}_batches.json", "w") as f:
            json.dump(batches, f)

        with open("tests/tmp/rank_0_batches.json") as f:
            rank_0_batches = torch.tensor(json.load(f))

        with open("tests/tmp/rank_1_batches.json") as f:
            rank_1_batches = torch.tensor(json.load(f))

        samples = [i.item() for item in zip(rank_0_batches.flatten(), rank_1_batches.flatten()) for i in item]

        assert samples == (list(range(num_samples)) * 3)[4:]
