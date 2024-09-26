from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import BatchSampler, SequentialSampler

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.pydanctic_if_types import PydanticLLMDataLoaderIFType
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.dataset import Dataset
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


class SequenceDataset(Dataset):
    def __init__(self, sequence: Sequence):
        super().__init__(raw_data_path=None, sample_key=None)
        self.sequence = sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, idx: int) -> Any:
        return self.sequence[idx]


def test_resumable_dataloader():
    batch_size = 3
    dataset = list(range(12))[::-1]
    seq_sampler = SequentialSampler(data_source=dataset)
    batch_sampler = BatchSampler(sampler=seq_sampler, batch_size=batch_size, drop_last=False)
    dataloader = LLMDataLoader(dataloader_tag="train", dataset=dataset, batch_sampler=batch_sampler)
    flat_samples = torch.cat([i for i in dataloader])
    original_samples = torch.IntTensor(dataset)
    assert (flat_samples == original_samples).all()


def test_dataloader_batching():
    batch_size = 2
    dataset = list(range(10))
    seq_sampler = SequentialSampler(data_source=dataset)
    batch_sampler = BatchSampler(sampler=seq_sampler, batch_size=batch_size, drop_last=False)
    dataloader = LLMDataLoader(dataloader_tag="train", dataset=dataset, batch_sampler=batch_sampler)

    batches_1 = torch.stack([i for i in dataloader])
    batches_2 = torch.stack([i for i in dataloader])
    assert batches_1.equal(batches_2)

    assert batches_1.flatten().tolist() == dataset


def test_skipped_and_distributed_dataloader_from_config():
    class DataloaderTestModel(BaseModel):
        train_dataloader: PydanticLLMDataLoaderIFType
        skip_num_samples: int

    root_dir = Path(__file__).parents[0]

    config_path = root_dir / "yaml_configs/skipped_dataloader.yaml"
    config_dict = load_app_config_dict(config_path)

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    components_rank_0: DataloaderTestModel = component_factory.build_components(
        config_dict=config_dict, components_model_type=DataloaderTestModel
    )

    world_size = config_dict["settings"]["cuda_env"]["world_size"]
    local_micro_batch_size = config_dict["settings"]["training"]["local_train_micro_batch_size"]
    skip_num_local_batches = components_rank_0.skip_num_samples // world_size // local_micro_batch_size

    assert world_size == 2
    assert skip_num_local_batches == 2

    config_dict["settings"]["cuda_env"]["global_rank"] = 1
    config_dict["train_dataloader"]["config"]["batch_sampler"]["config"]["sampler"]["config"]["rank"] = 1
    components_rank_1: DataloaderTestModel = component_factory.build_components(
        config_dict=config_dict, components_model_type=DataloaderTestModel
    )

    dataset = components_rank_0.train_dataloader.dataset

    batches_rank_0 = [batch for batch in components_rank_0.train_dataloader]
    batches_rank_1 = [batch for batch in components_rank_1.train_dataloader]

    # make sure that the dataloaders for the two ranks have the correct number of batches
    assert (
        len(components_rank_0.train_dataloader)
        == (len(dataset) - components_rank_0.skip_num_samples) // world_size // local_micro_batch_size
    )
    assert (
        len(components_rank_1.train_dataloader)
        == (len(dataset) - components_rank_1.skip_num_samples) // world_size // local_micro_batch_size
    )

    # we manually build up the batches from each dataloader to compare on a value basis
    # with [skip_num_local_batches:] we skip the first two batches
    dataset_indices_rank_0 = np.arange(0, 28, 2).reshape(-1, local_micro_batch_size)[skip_num_local_batches:]
    dataset_indices_rank_1 = np.arange(1, 29, 2).reshape(-1, local_micro_batch_size)[skip_num_local_batches:]

    assert np.all((dataset_indices_rank_0 == list(components_rank_0.train_dataloader.batch_sampler)))
    assert np.all((dataset_indices_rank_1 == list(components_rank_1.train_dataloader.batch_sampler)))

    batches_recomputed_rank_0 = []
    for batch_indices in dataset_indices_rank_0:
        sampled_pair = [torch.tensor(dataset[idx]["input_ids"]) for idx in batch_indices]
        # we stack the two samples into a batch and remove the last token from each sample
        # to get a proper "training" sample without the final target
        samples_pair_tensor = torch.stack(sampled_pair, dim=1).transpose(0, 1)[:, :-1]
        batches_recomputed_rank_0.append(samples_pair_tensor)

    batches_recomputed_rank_1 = []
    for batch_indices in dataset_indices_rank_1:
        sampled_pair = [torch.tensor(dataset[idx]["input_ids"]) for idx in batch_indices]
        # we stack the two samples into a batch and remove the last token from each sample
        # to get a proper "training" sample without the final target
        samples_pair_tensor = torch.stack(sampled_pair, dim=1).transpose(0, 1)[:, :-1]
        batches_recomputed_rank_1.append(samples_pair_tensor)

    for batch_1, batch_2 in zip(batches_rank_0, batches_recomputed_rank_0):
        assert (batch_1.samples["input_ids"] == batch_2).all()

    for batch_1, batch_2 in zip(batches_rank_1, batches_recomputed_rank_1):
        assert (batch_1.samples["input_ids"] == batch_2).all()

    for batch_1, batch_2 in zip(batches_rank_0, batches_rank_1):
        assert ~(batch_1.samples["input_ids"] == batch_2.samples["input_ids"]).all()
        assert ~(batch_1.targets["target_ids"] == batch_2.targets["target_ids"]).all()
