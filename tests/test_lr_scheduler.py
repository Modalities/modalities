from unittest.mock import call

from modalities.gym import Gym
from tests.test_utils import configure_dataloader_mock


def test_run_scheduler(
    monkeypatch,
    checkpointing_mock,
    evaluator_mock,
    nn_model_mock,
    optimizer_mock,
    scheduler_mock,
    loss_mock,
    llm_data_loader_mock,
    set_env_cpu,
    trainer,
):
    num_batches = 4
    num_ranks = 1

    llm_data_loader_mock, batches = configure_dataloader_mock(
        batch_size=32,
        seq_len=64,
        num_batches=num_batches,
        sample_key="input_ids",
        target_key="target_ids",
        llm_data_loader_mock=llm_data_loader_mock,
    )

    gym = Gym(trainer=trainer, evaluator=evaluator_mock, loss_fun=loss_mock, num_ranks=num_ranks)
    gym.run(
        model=nn_model_mock,
        optimizer=optimizer_mock,
        scheduler=scheduler_mock,
        callback_interval_in_batches=int(num_batches),
        train_data_loader=llm_data_loader_mock,
        evaluation_data_loaders=[],
        checkpointing=checkpointing_mock,
    )
    nn_model_mock.forward.assert_has_calls([call(b.samples) for b in batches])
    scheduler_mock.step.assert_called()
