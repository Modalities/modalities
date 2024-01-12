import torch
from torch.utils.data.sampler import BatchSampler

from modalities.dataloader.samplers import ResumableBatchSampler


def test_resumable_sampler(resumable_batch_sampler: ResumableBatchSampler):
    existing_sampler: BatchSampler = resumable_batch_sampler.underlying_batch_sampler
    indices_1 = [i for i in resumable_batch_sampler]
    indices_2 = [i for i in existing_sampler][resumable_batch_sampler.start_index :]

    data_source = existing_sampler.sampler.data_source[resumable_batch_sampler.start_index :]
    assert indices_1 == indices_2
    assert indices_1 != data_source


def test_resumable_batch_sampler(resumable_batch_sampler: ResumableBatchSampler):
    underlying_batch_sampler: BatchSampler = resumable_batch_sampler.underlying_batch_sampler
    values_1 = [i for i in resumable_batch_sampler]

    values_2_flat = underlying_batch_sampler.sampler.data_source[::-1][
        underlying_batch_sampler.batch_size * resumable_batch_sampler.start_index :
    ]
    values_2 = torch.IntTensor(values_2_flat).reshape([-1, underlying_batch_sampler.batch_size]).tolist()
    assert values_1 == values_2
