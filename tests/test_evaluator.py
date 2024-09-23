from unittest.mock import call

import torch

from modalities.batch import DatasetBatch
from modalities.evaluator import Evaluator


def test_evaluate_cpu(
    monkeypatch, nn_model_mock, loss_mock, llm_data_loader_mock, progress_publisher_mock, set_env_cpu
):
    batch_size = 32
    seq_len = 64
    num_batches = 4
    sample_key = "input_ids"
    target_key = "target_ids"

    sample_tensor = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    samples = {sample_key: sample_tensor[:, :-1]}
    targets = {target_key: sample_tensor[:, 1:]}

    batches = [DatasetBatch(targets=targets, samples=samples) for _ in range(num_batches)]

    llm_data_loader_mock.__iter__ = lambda _: iter(batches)
    llm_data_loader_mock.batch_size = batch_size

    evaluator = Evaluator(
        progress_publisher=progress_publisher_mock,
        evaluation_result_publisher=progress_publisher_mock,
    )

    evaluator.evaluate(
        model=nn_model_mock, data_loaders=[llm_data_loader_mock], loss_fun=loss_mock, num_train_steps_done=1
    )
    nn_model_mock.forward.assert_has_calls([call(b.samples) for b in batches])
