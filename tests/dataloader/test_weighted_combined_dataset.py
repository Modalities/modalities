from collections import Counter
from pathlib import Path

import pytest
from pydantic import BaseModel

from modalities.config.component_factory import ComponentFactory
from modalities.config.pydantic_if_types import PydanticDatasetIFType
from modalities.dataloader.dataset import WeightedCombinedDataset
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry


class _RangeDataset:
    """Minimal dataset returning (tag, index), so samples can be traced to their source."""

    def __init__(self, num_samples: int, tag: str):
        self._num_samples = num_samples
        self._tag = tag

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self._tag, idx


@pytest.fixture
def datasets() -> list[_RangeDataset]:
    return [_RangeDataset(100, "a"), _RangeDataset(50, "b"), _RangeDataset(10, "c")]


def test_integer_repeat_factors_repeat_whole_datasets(datasets):
    dataset = WeightedCombinedDataset(datasets, repeat_factors=[1.0, 2.0, 3.0])

    assert len(dataset) == 100 + 100 + 30
    counts = Counter(tag for tag, _ in (dataset[i] for i in range(len(dataset))))
    assert counts == {"a": 100, "b": 100, "c": 30}


def test_fractional_repeat_factor_adds_a_partial_pass(datasets):
    dataset = WeightedCombinedDataset(datasets, repeat_factors=[0.0, 2.5, 0.0])

    assert len(dataset) == 125
    drawn = [idx for _, idx in (dataset[i] for i in range(len(dataset)))]
    counts = Counter(drawn)
    # Two full passes plus half a pass: every document twice, half of them three times.
    assert set(counts) == set(range(50))
    assert sorted(counts.values()) == [2] * 25 + [3] * 25


def test_partial_pass_selects_distinct_documents(datasets):
    dataset = WeightedCombinedDataset(datasets, repeat_factors=[0.3, 0.0, 0.0])

    drawn = [idx for _, idx in (dataset[i] for i in range(len(dataset)))]
    assert len(drawn) == 30
    assert len(set(drawn)) == 30, "a partial pass must not draw the same document twice"


def test_downsampling_spreads_across_the_dataset(datasets):
    # A prefix would over-sample whatever the corpus is ordered by, so the partial pass
    # must reach into the whole index range rather than the front of it.
    dataset = WeightedCombinedDataset(datasets, repeat_factors=[0.1, 0.0, 0.0])

    drawn = sorted(idx for _, idx in (dataset[i] for i in range(len(dataset))))
    assert len(drawn) == 10
    assert max(drawn) > 50, f"partial pass stayed in the front of the dataset: {drawn}"


def test_repeat_factor_rounding_up_becomes_a_full_pass(datasets):
    dataset = WeightedCombinedDataset(datasets, repeat_factors=[0.999, 0.0, 0.0])

    assert len(dataset) == 100
    drawn = [idx for _, idx in (dataset[i] for i in range(len(dataset)))]
    assert sorted(drawn) == list(range(100))


def test_zero_repeat_factor_excludes_a_dataset(datasets):
    dataset = WeightedCombinedDataset(datasets, repeat_factors=[1.0, 0.0, 1.0])

    assert len(dataset) == 110
    counts = Counter(tag for tag, _ in (dataset[i] for i in range(len(dataset))))
    assert "b" not in counts


def test_same_seed_gives_the_same_blend(datasets):
    first = WeightedCombinedDataset(datasets, repeat_factors=[1.5, 0.4, 1.0], seed=7)
    second = WeightedCombinedDataset(datasets, repeat_factors=[1.5, 0.4, 1.0], seed=7)

    assert [first[i] for i in range(len(first))] == [second[i] for i in range(len(second))]


def test_different_seed_changes_the_partial_pass(datasets):
    first = WeightedCombinedDataset(datasets, repeat_factors=[0.5, 0.0, 0.0], seed=7)
    second = WeightedCombinedDataset(datasets, repeat_factors=[0.5, 0.0, 0.0], seed=8)

    assert {idx for _, idx in (first[i] for i in range(len(first)))} != {
        idx for _, idx in (second[i] for i in range(len(second)))
    }


def test_out_of_bounds_index_raises(datasets):
    dataset = WeightedCombinedDataset(datasets, repeat_factors=[1.0, 1.0, 1.0])

    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_negative_index_counts_from_the_end(datasets):
    dataset = WeightedCombinedDataset(datasets, repeat_factors=[1.0, 1.0, 1.0])

    assert dataset[-1] == dataset[len(dataset) - 1]


def test_mismatched_repeat_factors_are_rejected(datasets):
    with pytest.raises(ValueError, match="repeat factors"):
        WeightedCombinedDataset(datasets, repeat_factors=[1.0, 1.0])


def test_negative_repeat_factor_is_rejected(datasets):
    with pytest.raises(ValueError, match="non-negative"):
        WeightedCombinedDataset(datasets, repeat_factors=[1.0, -1.0, 1.0])


class _DatasetOnlyModel(BaseModel):
    train_dataset: PydanticDatasetIFType


def test_weighted_combined_is_buildable_from_a_config(dummy_packed_data_path: Path):
    # Guards the registry wiring: component key, variant key, config model and factory
    # signature all have to line up for a config to resolve.
    config_dict = {
        "train_dataset": {
            "component_key": "dataset",
            "variant_key": "weighted_combined",
            "config": {
                "datasets": [
                    {
                        "component_key": "dataset",
                        "variant_key": "packed_mem_map_dataset_continuous",
                        "config": {
                            "raw_data_path": str(dummy_packed_data_path),
                            "sequence_length": 4,
                            "sample_key": "input_ids",
                            "reuse_last_target": True,
                        },
                    }
                ],
                "repeat_factors": [2.0],
                "seed": 3,
            },
        }
    }

    component_factory = ComponentFactory(registry=Registry(COMPONENTS))
    components = component_factory.build_components(config_dict=config_dict, components_model_type=_DatasetOnlyModel)

    dataset = components.train_dataset
    assert isinstance(dataset, WeightedCombinedDataset)
    assert dataset.repeat_factors == [2.0]
    assert len(dataset) == 2 * len(dataset.datasets[0])


def test_negative_repeat_factor_is_rejected_by_the_config_model():
    from modalities.config.config import WeightedCombinedDatasetConfig

    with pytest.raises(ValueError):
        WeightedCombinedDatasetConfig(datasets=[], repeat_factors=[-1.0])
