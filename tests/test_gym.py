from unittest.mock import call

from modalities.gym import Gym
from tests.test_utils import configure_dataloader_mock


def test_run_cpu_only(
    monkeypatch,
    checkpoint_saving_mock,
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
        training_log_interval_in_steps=1,
        checkpointing_interval_in_steps=1,
        evaluation_interval_in_steps=1,
        train_data_loader=llm_data_loader_mock,
        evaluation_data_loaders=[],
        checkpoint_saving=checkpoint_saving_mock,
    )
    nn_model_mock.forward.assert_has_calls([call(b.samples) for b in batches])
    optimizer_mock.step.assert_called()
