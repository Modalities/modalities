import os
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock, call
from modalities.dataloader.dataloader import LLMDataLoader

import pytest
import torch

from modalities.batch import DatasetBatch
from modalities.evaluation.measure import AggregativeMeasure, AggregativeMeasureFactory
from modalities.evaluation.throughput_aggregator import ThroughputAggregator
from modalities.evaluator import Evaluator
from tests.conftest import set_env_cpu


class MockAggregativeMeasureFactory(AggregativeMeasureFactory):
    def __init__(self):
        self.created_mocks = []

    def create(self, local_rank: int) -> AggregativeMeasure:
        measure_mock = MagicMock(spec=AggregativeMeasure)
        measure_mock.compute.return_value = 0.0
        self.created_mocks.append(measure_mock)
        return measure_mock


@pytest.fixture
def throughput_aggregator():
    return MagicMock(spec=ThroughputAggregator)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_evaluate_cpu(nn_model_mock, llm_data_loader_mock, progress_publisher_mock, throughput_aggregator):
    batches = _prepare_test_batches(llm_data_loader_mock)

    measure_factory = MockAggregativeMeasureFactory()

    evaluator = Evaluator(
        local_rank=int(os.getenv("LOCAL_RANK")),
        loss_factories=[measure_factory],
        metric_factories=[measure_factory],
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        throughput_aggregator_factory=lambda: throughput_aggregator,
    )

    evaluator.evaluate(
        model=nn_model_mock,
        data_loaders=[llm_data_loader_mock],
        global_train_sample_id=0,
        local_sample_id_to_global_sample_id=lambda i: i,
    )
    nn_model_mock.forward.assert_has_calls([call(b.samples) for b in batches])


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_evaluate_builds_in_all_loss_factories(
    nn_model_mock, llm_data_loader_mock, llm_data_loader_mock2, progress_publisher_mock, throughput_aggregator
):
    num_batches = 4
    num_batches2 = 3
    loss_factories, _ = _run_loss_and_metrics_test(
        num_batches,
        llm_data_loader_mock,
        llm_data_loader_mock2,
        nn_model_mock,
        progress_publisher_mock,
        throughput_aggregator,
        num_batches2=num_batches2,
    )
    expected_num_calls = len(loss_factories) * (2 if num_batches2 > 0 else 1)
    assert len(loss_factories[0].created_mocks) == expected_num_calls


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_evaluate_calls_add_for_all_losses_and_batches(
    nn_model_mock, llm_data_loader_mock, llm_data_loader_mock2, progress_publisher_mock, throughput_aggregator
):
    num_batches = 4
    num_batches2 = 3
    loss_factories, _ = _run_loss_and_metrics_test(
        num_batches,
        llm_data_loader_mock,
        llm_data_loader_mock2,
        nn_model_mock,
        progress_publisher_mock,
        throughput_aggregator,
        num_batches2=num_batches2,
    )
    add_call_counts = [loss.add.call_count for loss in loss_factories[0].created_mocks]
    expected_call_counts = [num_batches] * len(loss_factories) + [num_batches2] * len(loss_factories)
    unittest.TestCase().assertListEqual(add_call_counts, expected_call_counts)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_evaluate_calls_compute_on_all_losses(
    nn_model_mock, llm_data_loader_mock, llm_data_loader_mock2, progress_publisher_mock, throughput_aggregator
):
    num_batches = 4
    num_batches2 = 3
    loss_factories, _ = _run_loss_and_metrics_test(
        num_batches,
        llm_data_loader_mock,
        llm_data_loader_mock2,
        nn_model_mock,
        progress_publisher_mock,
        throughput_aggregator,
        num_batches2=num_batches2,
    )
    compute_call_counts = [loss.compute.call_count for loss in loss_factories[0].created_mocks]
    expected_call_counts = [1] * len(loss_factories) * (2 if num_batches2 > 0 else 1)
    unittest.TestCase().assertListEqual(compute_call_counts, expected_call_counts)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_evaluate_builds_in_all_metrics_factories(
    nn_model_mock, llm_data_loader_mock, llm_data_loader_mock2, progress_publisher_mock, throughput_aggregator
):
    num_batches = 4
    num_batches2 = 3
    _, metric_factories = _run_loss_and_metrics_test(
        num_batches,
        llm_data_loader_mock,
        llm_data_loader_mock2,
        nn_model_mock,
        progress_publisher_mock,
        throughput_aggregator,
        num_batches2=num_batches2,
    )
    expected_num_calls = len(metric_factories) * (2 if num_batches2 > 0 else 1)
    assert len(metric_factories[0].created_mocks) == expected_num_calls


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_evaluate_calls_add_for_all_metrics_and_batches(
    nn_model_mock, llm_data_loader_mock, llm_data_loader_mock2, progress_publisher_mock, throughput_aggregator
):
    num_batches = 4
    num_batches2 = 3
    _, metric_factories = _run_loss_and_metrics_test(
        num_batches,
        llm_data_loader_mock,
        llm_data_loader_mock2,
        nn_model_mock,
        progress_publisher_mock,
        throughput_aggregator,
        num_batches2=num_batches2,
    )
    add_call_counts = [loss.add.call_count for loss in metric_factories[0].created_mocks]
    expected_call_counts = [num_batches] * len(metric_factories) + [num_batches2] * len(metric_factories)
    unittest.TestCase().assertListEqual(add_call_counts, expected_call_counts)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_evaluate_calls_compute_on_all_metrics(
    nn_model_mock, llm_data_loader_mock, llm_data_loader_mock2, progress_publisher_mock, throughput_aggregator
):
    num_batches = 4
    num_batches2 = 3
    _, metric_factories = _run_loss_and_metrics_test(
        num_batches,
        llm_data_loader_mock,
        llm_data_loader_mock2,
        nn_model_mock,
        progress_publisher_mock,
        throughput_aggregator,
        num_batches2=num_batches2,
    )
    compute_call_counts = [loss.compute.call_count for loss in metric_factories[0].created_mocks]
    expected_call_counts = [1] * len(metric_factories) * (2 if num_batches2 > 0 else 1)
    unittest.TestCase().assertListEqual(compute_call_counts, expected_call_counts)


def _run_loss_and_metrics_test(
    num_batches: int,
    llm_data_loader_mock: LLMDataLoader,
    llm_data_loader_mock2: LLMDataLoader,
    nn_model_mock,
    progress_publisher_mock,
    throughput_aggregator,
    num_batches2: int = 0,
) -> Tuple[List[MockAggregativeMeasureFactory], List[MockAggregativeMeasureFactory]]:
    loss_factory = MockAggregativeMeasureFactory()
    metric_factory = MockAggregativeMeasureFactory()
    loss_factories = [loss_factory] * 3
    metric_factories = [metric_factory] * 5

    data_loaders = _prepare_data_loaders(num_batches, num_batches2, llm_data_loader_mock, llm_data_loader_mock2)

    evaluator = Evaluator(
        local_rank=int(os.getenv("LOCAL_RANK", 0)),
        loss_factories=loss_factories,
        metric_factories=metric_factories,
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
        throughput_aggregator_factory=lambda: throughput_aggregator,
    )

    evaluator.evaluate(
        model=nn_model_mock,
        data_loaders=data_loaders,
        global_train_sample_id=0,
        local_sample_id_to_global_sample_id=lambda i: i,
    )
    return loss_factories, metric_factories


def _prepare_data_loaders(
    num_batches: int, num_batches2: int, llm_data_loader_mock: LLMDataLoader, llm_data_loader_mock2: LLMDataLoader
):
    _prepare_test_batches(llm_data_loader_mock, num_batches=num_batches)
    data_loaders = [llm_data_loader_mock]

    if num_batches2 > 0:
        _prepare_test_batches(llm_data_loader_mock2, batch_size=17, seq_len=13, num_batches=num_batches2)
        data_loaders.append(llm_data_loader_mock2)
    return data_loaders


def _prepare_test_batches(
    llm_data_loader_mock: MagicMock,
    batch_size: int = 32,
    seq_len: int = 64,
    num_batches: int = 4,
    sample_key: str = "input_ids",
    target_key: str = "target_ids",
) -> List[DatasetBatch]:
    sample_tensor = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    samples = {sample_key: sample_tensor[:, :-1]}
    targets = {target_key: sample_tensor[:, 1:]}

    batches = [DatasetBatch(targets=targets, samples=samples) for _ in range(num_batches)]

    llm_data_loader_mock.__iter__ = lambda _: iter(batches)
    llm_data_loader_mock.drop_last = False
    llm_data_loader_mock.batch_size = batch_size

    llm_data_loader_mock.dataset = MagicMock()

    return batches
