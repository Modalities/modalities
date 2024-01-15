import os
from typing import List
from unittest.mock import MagicMock, call

import torch

from modalities.batch import DatasetBatch
from modalities.evaluator import Evaluator
from modalities.loss_functions import Loss
from modalities.metrics import Metric


def test_evaluate_cpu(
    monkeypatch, nn_model_mock, loss_mock, metric_mock, llm_data_loader_mock, progress_publisher_mock, set_env_cpu
):
    batches = _prepare_test_batches(llm_data_loader_mock)

    evaluator = Evaluator(
        local_rank=int(os.getenv("LOCAL_RANK")),
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
    )

    evaluator.evaluate(
        model=nn_model_mock,
        data_loaders=[llm_data_loader_mock],
        loss_functions=[loss_mock],
        metrics=[metric_mock],
        global_train_sample_id=0,
        local_sample_id_to_global_sample_id=lambda i: i,
    )
    nn_model_mock.forward.assert_has_calls([call(b.samples) for b in batches])


def test_evaluate_calls_all_losses(
    monkeypatch, nn_model_mock, loss_mock, metric_mock, llm_data_loader_mock, progress_publisher_mock, set_env_cpu
):
    _prepare_test_batches(llm_data_loader_mock)

    evaluator = Evaluator(
        local_rank=int(os.getenv("LOCAL_RANK")),
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
    )

    other_loss_mock = MagicMock(spec=Loss, return_value=torch.rand(1, requires_grad=True))

    evaluator.evaluate(
        model=nn_model_mock,
        data_loaders=[llm_data_loader_mock],
        loss_functions=[loss_mock, loss_mock, other_loss_mock],
        metrics=[metric_mock],
        global_train_sample_id=0,
        local_sample_id_to_global_sample_id=lambda i: i,
    )
    assert loss_mock.call_count == 4 * 2


def test_evaluate_calls_all_metrics(
    monkeypatch, nn_model_mock, loss_mock, metric_mock, llm_data_loader_mock, progress_publisher_mock, set_env_cpu
):
    _prepare_test_batches(llm_data_loader_mock)

    evaluator = Evaluator(
        local_rank=int(os.getenv("LOCAL_RANK")),
        batch_progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
    )

    other_metric_mock = MagicMock(spec=Metric, return_value=torch.rand(1, requires_grad=True))

    evaluator.evaluate(
        model=nn_model_mock,
        data_loaders=[llm_data_loader_mock],
        loss_functions=[loss_mock],
        metrics=[metric_mock, metric_mock, other_metric_mock],
        global_train_sample_id=0,
        local_sample_id_to_global_sample_id=lambda i: i,
    )
    assert metric_mock.call_count == 4 * 2


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
    llm_data_loader_mock.batch_size = batch_size

    return batches
