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
    # we test that drop_last and or reusing the initial samples works as expected
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
        # so that every data parallel ran has a full last batch
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
    # we test that shuffling leads to a different order of the samples and all samples of the
    # original dataset are used
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
    assert samples_flat != dataset


@pytest.mark.parametrize(
    "num_samples, epoch, shuffle, seed, drop_last, skip_num_global_samples",
    [
        (30, 0, False, 0, True, 0),
        (30, 0, True, 0, True, 0),
    ],
)
def test_ordering_with_different_world_sizes_and_shuffling(
    num_samples: int, epoch: int, shuffle: bool, seed: int, drop_last: bool, skip_num_global_samples: int
):
    # 1) we test that WITHOUT shuffling the order of samples is the same as in the original dataset
    # for different world sizes.
    # 2) we test that WITH shuffling the order of samples is the same for different world sizes
    # but not the same order as in the original dataset.
    dataset = list(range(num_samples))
    samplers_3 = [
        ResumableDistributedSampler(
            dataset=dataset,
            rank=rank,
            num_replicas=3,
            epoch=epoch,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            skip_num_global_samples=skip_num_global_samples,
        )
        for rank in range(3)
    ]

    samplers_6 = [
        ResumableDistributedSampler(
            dataset=dataset,
            rank=rank,
            num_replicas=6,
            epoch=epoch,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            skip_num_global_samples=skip_num_global_samples,
        )
        for rank in range(6)
    ]

    samples_3 = [[dataset[i] for i in sampler] for sampler in samplers_3]
    samples_flat_3 = [s for t in zip(*samples_3) for s in t]

    samples_6 = [[dataset[i] for i in sampler] for sampler in samplers_6]
    samples_flat_6 = [s for t in zip(*samples_6) for s in t]

    if not shuffle:
        assert dataset == samples_flat_3
        assert dataset == samples_flat_6
    else:
        assert samples_flat_3 == samples_flat_6
        assert set(samples_flat_3) == set(dataset)
        assert samples_flat_6 != dataset
