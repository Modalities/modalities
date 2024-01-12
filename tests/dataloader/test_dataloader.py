import torch
from torch.utils.data import BatchSampler, SequentialSampler

from modalities.config.config import AppConfig
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.dataloader_factory import DataloaderFactory
from modalities.dataloader.samplers import ResumableBatchSampler
from modalities.resolver_register import ResolverRegister


def test_resumable_dataloader() -> LLMDataLoader:
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


def test_dataloader_from_config(dummy_config: AppConfig):
    resolvers = ResolverRegister(config=dummy_config)
    start_index = 2
    dataloader_1: LLMDataLoader = DataloaderFactory.get_dataloader(
        resolvers=resolvers, config=dummy_config.data.train_dataloader, skip_num_batches=start_index
    )
    dataset = dataloader_1.dataset

    distributed_sampler = dataloader_1.batch_sampler.underlying_batch_sampler.sampler
    batch_sampler = BatchSampler(
        sampler=distributed_sampler, batch_size=dataloader_1.sampler_batch_size, drop_last=False
    )
    dataloader_2 = LLMDataLoader(
        dataloader_tag="train", dataset=dataset, batch_sampler=batch_sampler, collate_fn=dataloader_1.collate_fn
    )

    samples_1 = [batch for _, batch in zip(range(10), dataloader_1)]
    samples_2 = [batch for _, batch in zip(range(10), dataloader_2)]

    assert dataloader_1.sampler_batch_size * len(dataloader_2) == len(dataset)

    assert len(dataloader_1) + start_index == len(dataloader_2)

    for batch_1, batch_2 in zip(samples_1, samples_2):
        assert ~(batch_1.samples["input_ids"] == batch_2.samples["input_ids"]).all()
        assert ~(batch_1.targets["target_ids"] == batch_2.targets["target_ids"]).all()

    for batch_1, batch_2 in zip(samples_1, samples_2[start_index:]):
        assert (batch_1.samples["input_ids"] == batch_2.samples["input_ids"]).all()
        assert (batch_1.targets["target_ids"] == batch_2.targets["target_ids"]).all()
