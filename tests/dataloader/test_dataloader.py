from typing import Dict

import torch
from pydantic import BaseModel
from torch.utils.data import BatchSampler, SequentialSampler

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import PydanticLLMDataLoaderIFType
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
    dummy_config["train_dataloader"]["config"]["skip_num_steps"] = start_index

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
    repeating_dataloader = RepeatingDataLoader(dataloader=dataloader_skipped, reshuffle_after_epoch=False)

    num_batches_per_epoch = num_samples // batch_size
    # get the batches for two epochs
    batches_1 = torch.stack([i for i in dataloader_skipped] + [i for i in dataloader])
    batches_2 = torch.stack(
        [i for _, i in zip(range(num_batches_per_epoch * 2 - skip_num_batches), repeating_dataloader)]
    )

    assert batches_1.equal(batches_2)
    assert batches_1.flatten().tolist() == dataset[skip_num_batches * batch_size :] + dataset


def test_repeating_dataloader_with_shuffling():
    batch_size = 2
    skip_num_batches = 2
    num_samples = 10
    dataset = list(range(num_samples))
    seq_sampler = SequentialSampler(data_source=dataset)
    batch_sampler = BatchSampler(sampler=seq_sampler, batch_size=batch_size, drop_last=False)
    # the LLMDataLoader always requires a ResumableBatchSampler
    # create the dataloader that skips the first skip_num_batches
    resumable_batch_sampler_skipped = ResumableBatchSampler(
        underlying_batch_sampler=batch_sampler, start_index=skip_num_batches
    )
    dataloader_skipped = LLMDataLoader(
        dataloader_tag="train", dataset=dataset, batch_sampler=resumable_batch_sampler_skipped
    )

    # create dataloader that skips not batches
    resumable_batch_sampler = ResumableBatchSampler(underlying_batch_sampler=batch_sampler, start_index=0)
    dataloader = LLMDataLoader(dataloader_tag="train", dataset=dataset, batch_sampler=resumable_batch_sampler)

    # create repeating dataloader that first skips the skip_num_batches
    # in epoch 0 and then returns the batches from the beginning
    repeating_dataloader = RepeatingDataLoader(dataloader=dataloader, reshuffle_after_epoch=True)

    num_batches_per_epoch = num_samples // batch_size
    # get the batches for two epochs
    batches_1 = torch.stack([i for i in dataloader_skipped] + [i for i in dataloader])
    batches_2 = torch.stack(
        [i for _, i in zip(range(num_batches_per_epoch * 2 - skip_num_batches), repeating_dataloader)]
    )

    assert batches_1.equal(batches_2)
