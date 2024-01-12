from unittest.mock import call, patch

import torch

from modalities.batch import DatasetBatch
from modalities.running_env.fsdp.reducer import Reducer
from modalities.gym import Gym


def test_run_cpu_only(
    monkeypatch,
    checkpointing_mock,
    evaluator_mock,
    nn_model_mock,
    optimizer_mock,
    loss_mock,
    llm_data_loader_mock,
    set_env_cpu,
    trainer,
):
    batch_size = 32
    seq_len = 64
    num_batches = 4
    sample_key = "input_ids"
    target_key = "target_ids"
    num_ranks = 1

    sample_tensor = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    samples = {sample_key: sample_tensor[:, :-1]}
    targets = {target_key: sample_tensor[:, 1:]}

    batches = [DatasetBatch(targets=targets, samples=samples) for _ in range(num_batches)]

    llm_data_loader_mock.__iter__ = lambda _: iter(batches)
    llm_data_loader_mock.batch_size = batch_size
    llm_data_loader_mock.fast_forward_batch_id = 0

    gym = Gym(trainer=trainer, evaluator=evaluator_mock, loss_fun=loss_mock, num_ranks=num_ranks)
    with patch.object(Reducer, "reduce", return_value=None) as reduce_mock:
        gym.run(
            model=nn_model_mock,
            optimizer=optimizer_mock,
            callback_interval_in_batches=int(num_batches),
            train_data_loader=llm_data_loader_mock,
            evaluation_data_loaders=[],
            checkpointing=checkpointing_mock,
        )
        nn_model_mock.forward.assert_has_calls([call(b.samples) for b in batches])
        optimizer_mock.step.assert_called()
        reduce_mock.assert_called()
