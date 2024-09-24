import math

import pytest

from modalities.dataloader.samplers import ResumableDistributedSampler


@pytest.mark.parametrize(
    "num_samples, epoch, shuffle, seed, drop_last, skip_num_global_samples",
    [
        (30, 0, False, 0, False, 0),
        (30, 0, False, 0, True, 0),  # drop_last has  no effect because integer divisible
        (30, 0, False, 0, False, 9),
        (30, 0, False, 0, True, 9),  # drop_last has  no effect because integer divisible
        (30, 0, False, 0, True, 10),  # drop_last has an effect because not integer divisible
        (30, 0, False, 0, False, 10),  # we have to reuse the initial samples (1 sample)
    ],
)
def test_dropping_and_reusing(
    num_samples: int, epoch: int, shuffle: bool, seed: int, drop_last: bool, skip_num_global_samples: int
):
    dataset = list(range(num_samples))
    num_replicas = 3  # world size
    samplers = [
        ResumableDistributedSampler(
            dataset=dataset,
            rank=rank,
            num_replicas=num_replicas,
            epoch=epoch,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            skip_num_global_samples=skip_num_global_samples,
        )
        for rank in range(num_replicas)
    ]

    samples = [[dataset[i] for i in sampler] for sampler in samplers]

    if drop_last:
        # when drop_last true, we drop the last samples so that every data parallel rank
        # has the same number of samples.
        # Note that also means that the last, remaining samples (i.e., maximum num_ranks -1)
        # are not used at all
        cut_off_samples = len(dataset) - (len(dataset) - skip_num_global_samples) % num_replicas
        padded_samples = []
    else:
        cut_off_samples = len(dataset)
        samples_left = len(dataset) - skip_num_global_samples
        padding_size = math.ceil(samples_left / num_replicas) * num_replicas - samples_left
        # when drop_last false, we reuse the last samples (i.e., maximum num_ranks -1)
        # so that every data parallel ran, has a full last batch
        padded_samples = dataset[:padding_size]

    assert dataset[skip_num_global_samples:cut_off_samples] + padded_samples == list(
        s for t in zip(*samples) for s in t
    )


@pytest.mark.parametrize(
    "num_samples, epoch, shuffle, seed, drop_last, skip_num_global_samples",
    [
        (30, 0, True, 0, True, 0),
    ],
)
def test_shuffling(
    num_samples: int, epoch: int, shuffle: bool, seed: int, drop_last: bool, skip_num_global_samples: int
):
    dataset = list(range(num_samples))
    num_replicas = 3  # world size
    samplers = [
        ResumableDistributedSampler(
            dataset=dataset,
            rank=rank,
            num_replicas=num_replicas,
            epoch=epoch,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            skip_num_global_samples=skip_num_global_samples,
        )
        for rank in range(num_replicas)
    ]

    samples = [[dataset[i] for i in sampler] for sampler in samplers]
    samples_flat = [s for t in zip(*samples) for s in t]

    assert set(samples_flat) == set(dataset)
