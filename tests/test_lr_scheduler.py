from unittest.mock import call

import numpy as np

from modalities.gym import Gym
from modalities.optimizers.lr_schedulers import DummyLRScheduler
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


def test_dummy_lr_scheduler(monkeypatch, optimizer_with_param_groups_mock):
    # we test that the optimizer step function reduces the lr by 0.01 for each param group.
    # we also test that the scheduler step function does not change the lr.

    scheduler = DummyLRScheduler(optimizer=optimizer_with_param_groups_mock)
    assert scheduler.get_lr() == [0.1, 0.2, 0.3]
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert scheduler.get_last_lr() == [0.1, 0.2, 0.3]

    optimizer_with_param_groups_mock.step()
    assert np.sum(np.abs(np.array(scheduler.get_lr()) - np.array([0.09, 0.19, 0.29]))) < 1e-6
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert scheduler.get_last_lr() == [0.1, 0.2, 0.3]

    scheduler.step()
    assert np.sum(np.abs(np.array(scheduler.get_lr()) - np.array([0.09, 0.19, 0.29]))) < 1e-6
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert np.sum(np.abs(np.array(scheduler.get_last_lr()) - np.array([0.09, 0.19, 0.29]))) < 1e-6

    optimizer_with_param_groups_mock.step()
    assert np.sum(np.abs(np.array(scheduler.get_lr()) - np.array([0.08, 0.18, 0.28]))) < 1e-6
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert np.sum(np.abs(np.array(scheduler.get_last_lr()) - np.array([0.09, 0.19, 0.29]))) < 1e-6

    scheduler.step()
    assert np.sum(np.abs(np.array(scheduler.get_lr()) - np.array([0.08, 0.18, 0.28]))) < 1e-6
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert np.sum(np.abs(np.array(scheduler.get_last_lr()) - np.array([0.08, 0.18, 0.28]))) < 1e-6
