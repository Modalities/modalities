import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from llm_gym.dataloader.samplers import ResumableSampler


def test_resumable_sampler(resumable_sampler: ResumableSampler):
    existing_sampler: SequentialSampler = resumable_sampler.existing_sampler
    values_1 = [i for i in resumable_sampler]
    values_2 = existing_sampler.data_source[::-1][resumable_sampler.start_index :]
    assert values_1 == values_2


def test_resumable_batch_sampler(resumable_batch_sampler: ResumableSampler):
    existing_sampler: BatchSampler = resumable_batch_sampler.existing_sampler
    values_1 = [i for i in resumable_batch_sampler]

    values_2_flat = existing_sampler.sampler.data_source[::-1][
        existing_sampler.batch_size * resumable_batch_sampler.start_index :
    ]
    values_2 = torch.IntTensor(values_2_flat).reshape([-1, existing_sampler.batch_size]).tolist()
    assert values_1 == values_2
