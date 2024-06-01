import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.pydanctic_if_types import PydanticLLMDataLoaderIFType
from modalities.dataloader.dataloader import LLMDataLoader, RepeatingDataLoader
from modalities.dataloader.samplers import ResumableBatchSampler
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


def test_resumable_dataloader():
    batch_size = 3
    start_index = 2
    dataset = list(range(12))[::-1]
    seq_sampler = SequentialSampler(data_source=dataset)
    batch_sampler = BatchSampler(sampler=seq_sampler, batch_size=batch_size, drop_last=False)
    resumable_batch_sampler = ResumableBatchSampler(underlying_batch_sampler=batch_sampler, start_index=start_index)
    dataloader = LLMDataLoader(dataloader_tag="train", dataset=dataset, batch_sampler=resumable_batch_sampler)
    flat_samples = torch.cat([i for i in dataloader])
    original_samples = torch.IntTensor(dataset[start_index * batch_size :])
    assert (flat_samples == original_samples).all()


def test_dataloader_from_config(dummy_config: Dict):
    start_index = 2
    dummy_config["train_dataloader"]["config"]["skip_num_batches"] = start_index

    class DataloaderTestModel(BaseModel):
        train_dataloader: PydanticLLMDataLoaderIFType

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    components: DataloaderTestModel = component_factory.build_components(
        config_dict=dummy_config, components_model_type=DataloaderTestModel
    )

    dataloader_1: LLMDataLoader = components.train_dataloader
    dataset = dataloader_1.dataset
    resumable_batch_sampler: ResumableBatchSampler = dataloader_1.batch_sampler
    distributed_sampler = resumable_batch_sampler.underlying_batch_sampler.sampler
    batch_sampler = BatchSampler(sampler=distributed_sampler, batch_size=dataloader_1.batch_size, drop_last=True)
    dataloader_2 = LLMDataLoader(
        dataloader_tag="train", dataset=dataset, batch_sampler=batch_sampler, collate_fn=dataloader_1.collate_fn
    )

    samples_1 = [batch for _, batch in zip(range(10), dataloader_1)]
    samples_2 = [batch for _, batch in zip(range(10), dataloader_2)]

    assert len(dataloader_2) == len(dataset) // dataloader_1.batch_size

    assert len(dataloader_1) + start_index == len(dataloader_2)

    for batch_1, batch_2 in zip(samples_1, samples_2):
        assert ~(batch_1.samples["input_ids"] == batch_2.samples["input_ids"]).all()
        assert ~(batch_1.targets["target_ids"] == batch_2.targets["target_ids"]).all()

    for batch_1, batch_2 in zip(samples_1, samples_2[start_index:]):
        assert (batch_1.samples["input_ids"] == batch_2.samples["input_ids"]).all()
        assert (batch_1.targets["target_ids"] == batch_2.targets["target_ids"]).all()


def test_dataloader_batching():
    batch_size = 2
    skip_num_batches = 2
    dataset = list(range(10))
    seq_sampler = SequentialSampler(data_source=dataset)
    batch_sampler = BatchSampler(sampler=seq_sampler, batch_size=batch_size, drop_last=False)
    # the LLMDataLoader always requires a ResumableBatchSampler
    resumable_batch_sampler = ResumableBatchSampler(
        underlying_batch_sampler=batch_sampler, start_index=skip_num_batches
    )
    dataloader = LLMDataLoader(dataloader_tag="train", dataset=dataset, batch_sampler=resumable_batch_sampler)

    batches_1 = torch.stack([i for i in dataloader])
    batches_2 = torch.stack([i for i in dataloader])
    assert batches_1.equal(batches_2)

    assert batches_1.flatten().tolist() == dataset[skip_num_batches * batch_size :]


def test_repeating_dataloader_without_shuffling():
    batch_size = 2
    skip_num_batches = 2
    num_samples = 10
    dataset = list(range(num_samples))
    seq_sampler = SequentialSampler(data_source=dataset)
    # the LLMDataLoader always requires a ResumableBatchSampler
    # create the dataloader that skips the first skip_num_batches
    batch_sampler_skipped = BatchSampler(sampler=seq_sampler, batch_size=batch_size, drop_last=True)
    resumable_batch_sampler_skipped = ResumableBatchSampler(
        underlying_batch_sampler=batch_sampler_skipped, start_index=skip_num_batches
    )
    dataloader_skipped = LLMDataLoader(
        dataloader_tag="train", dataset=dataset, batch_sampler=resumable_batch_sampler_skipped
    )

    # create dataloader that skips no batches
    batch_sampler = BatchSampler(sampler=seq_sampler, batch_size=batch_size, drop_last=True)
    resumable_batch_sampler = ResumableBatchSampler(underlying_batch_sampler=batch_sampler, start_index=0)
    dataloader = LLMDataLoader(dataloader_tag="train", dataset=dataset, batch_sampler=resumable_batch_sampler)

    # create repeating dataloader that first skips the skip_num_batches
    # in epoch 0 and then returns the batches from the beginning
    repeating_dataloader = RepeatingDataLoader(dataloader=dataloader_skipped, reshuffle_after_epoch=False, num_epochs=2)

    num_samples // batch_size
    # get the batches for two epochs
    batches_1 = torch.stack([i for i in dataloader_skipped] + [i for i in dataloader])
    batches_2 = torch.stack([i for i in repeating_dataloader])

    assert batches_1.equal(batches_2)
    assert batches_1.flatten().tolist() == dataset[skip_num_batches * batch_size :] + dataset


def test_repeating_dataloader_with_shuffling():
    batch_size = 2
    skip_num_batches = 2
    num_samples = 10
    dataset = list(range(num_samples))

    generator = torch.Generator().manual_seed(42)
    random_sampler = RandomSampler(data_source=dataset, generator=generator)
    batch_sampler = BatchSampler(sampler=random_sampler, batch_size=batch_size, drop_last=False)

    # create dataloader that skips not batches
    resumable_batch_sampler = ResumableBatchSampler(
        underlying_batch_sampler=batch_sampler, start_index=skip_num_batches
    )
    dataloader = LLMDataLoader(dataloader_tag="train", dataset=dataset, batch_sampler=resumable_batch_sampler)

    # create repeating dataloader that first skips the skip_num_batches
    # in epoch 0 and then returns the batches from the beginning
    repeating_dataloader = RepeatingDataLoader(dataloader=dataloader, reshuffle_after_epoch=False, num_epochs=2)

    # get the batches for two epochs
    num_batches_per_epoch = num_samples // batch_size
    batches = torch.stack([i for i in repeating_dataloader])
    batches_epoch_1 = batches[: num_batches_per_epoch - skip_num_batches]
    batches_epoch_2 = batches[num_batches_per_epoch - skip_num_batches :]
    # when we skip 2 batches only 3 batches are left, i.e., 6 samples
    assert len(set(batches_epoch_1.flatten().tolist())) == 6
    assert set(batches_epoch_2.flatten().tolist()) == set(range(10))


def test_skipped_and_distributed_dataloader_from_config():
    class DataloaderTestModel(BaseModel):
        train_dataloader: PydanticLLMDataLoaderIFType
        skip_num_batches: int

    root_dir = Path(__file__).parents[0]

    config_path = root_dir / "yaml_configs/skipped_dataloader.yaml"
    config_dict = load_app_config_dict(config_path)

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    components_rank_0: DataloaderTestModel = component_factory.build_components(
        config_dict=config_dict, components_model_type=DataloaderTestModel
    )

    config_dict["settings"]["cuda_env"]["global_rank"] = 1
    config_dict["train_dataloader"]["config"]["batch_sampler"]["config"]["sampler"]["config"]["rank"] = 1
    components_rank_1: DataloaderTestModel = component_factory.build_components(
        config_dict=config_dict, components_model_type=DataloaderTestModel
    )

    dataset = components_rank_0.train_dataloader.dataset

    batches_rank_0 = [batch for _, batch in zip(range(10), components_rank_0.train_dataloader)]
    batches_rank_1 = [batch for _, batch in zip(range(10), components_rank_1.train_dataloader)]

    # make sure that the dataloaders for the two ranks have the correct number of batches
    assert (
        len(components_rank_0.train_dataloader)
        == math.ceil(len(dataset) // 2) // components_rank_0.train_dataloader.batch_size
        - components_rank_0.skip_num_batches
    )
    assert (
        len(components_rank_1.train_dataloader)
        == math.ceil(len(dataset) // 2) // components_rank_0.train_dataloader.batch_size
        - components_rank_0.skip_num_batches
    )

    # we manually build up the batches from each dataloader to compare on a value basis
    # with [1:] we skip the first batch
    dataset_indices_rank_0 = np.arange(0, 28, 2).reshape(-1, 2)[1:]
    dataset_indices_rank_1 = np.arange(1, 29, 2).reshape(-1, 2)[1:]

    assert all((dataset_indices_rank_0 == list(components_rank_0.train_dataloader.batch_sampler)).flatten())
    assert all((dataset_indices_rank_1 == list(components_rank_1.train_dataloader.batch_sampler)).flatten())

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
