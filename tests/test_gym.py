from unittest.mock import call, patch

import torch

from llm_gym.batch import DatasetBatch
from llm_gym.fsdp.reducer import Reducer
from llm_gym.gym import Gym


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

    sample_tensor = torch.randint(size=(batch_size, seq_len), low=1, high=100)
    samples = {sample_key: sample_tensor[:, :-1]}
    targets = {target_key: sample_tensor[:, 1:]}

    batches = [DatasetBatch(targets=targets, samples=samples) for _ in range(num_batches)]

    llm_data_loader_mock.__iter__ = lambda _: iter(batches)

    gym = Gym(
        trainer=trainer,
        evaluator=evaluator_mock,
        loss_fun=loss_mock,
    )
    with patch.object(Reducer, "reduce", return_value=None) as reduce_mock:
        gym.run(
            model=nn_model_mock,
            optimizer=optimizer_mock,
            train_data_loader=llm_data_loader_mock,
            num_batches_per_rank=num_batches,
            evaluation_data_loaders=[],
            callback_interval_in_batches=int(num_batches),
            checkpointing=checkpointing_mock,
        )
        nn_model_mock.forward.assert_has_calls([call(b.samples) for b in batches])
        optimizer_mock.step.assert_called()
        reduce_mock.assert_called()
