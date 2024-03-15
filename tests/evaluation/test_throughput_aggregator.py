import pytest

from modalities.evaluation.throughput_aggregator import (
    ThroughputAggregationContext,
    ThroughputAggregator,
    start_throughput_measurement,
)
from modalities.util import TimeRecorderStates
from tests.conftest import set_env_cpu


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_computes_samples_per_second(capsys):
    with capsys.disabled():
        agg = ThroughputAggregator()
        agg.start()
        agg.stop(6)
        agg._recorder.delta_t = 3.0

        assert agg.compute_samples_per_second(0).item() == pytest.approx(2.0, 0.00001)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_reset():
    agg1 = ThroughputAggregator()
    agg1.start()
    agg1.stop(6)
    agg1.reset()

    assert agg1._num_samples == ThroughputAggregator()._num_samples
    assert agg1._recorder.delta_t == ThroughputAggregator()._recorder.delta_t


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_throughput_iterator_aggregates_all_added_samples():
    input_iterable = range(5)
    for agg, i in start_throughput_measurement(input_iterable):
        agg.stop(i)
    agg._recorder.delta_t = 1.0

    assert agg.compute_samples_per_second(0).item() == pytest.approx(sum(range(5)), 0.00001)


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_throughput_context_sets_correct_number_of_samples():
    with ThroughputAggregationContext(11, 0) as context:
        pass

    assert context._agg._num_samples == 11


@pytest.mark.usefixtures(set_env_cpu.__name__)
def test_throughput_context_stops_aggregation():
    with ThroughputAggregationContext(11, 0) as context:
        pass

    assert not context._agg._recorder._state == TimeRecorderStates.RUNNING
