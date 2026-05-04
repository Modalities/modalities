from unittest.mock import MagicMock, call

import numpy as np
import torch

from modalities.checkpointing.checkpoint_saving import CheckpointSaving
from modalities.checkpointing.stateful.app_state import AppState
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.evaluator import Evaluator
from modalities.gym import Gym
from modalities.loss_functions import Loss
from modalities.optimizers.lr_schedulers import DummyLRScheduler, LRSchedulerFactory
from modalities.trainer import Trainer
from tests.utility import configure_dataloader_mock


def test_run_scheduler(
    set_env_cpu,
    checkpoint_saving_mock: CheckpointSaving,
    evaluator_mock: Evaluator,
    app_state_mock: AppState,
    loss_mock: Loss,
    llm_data_loader_mock: LLMDataLoader,
    trainer: Trainer,
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
        app_state=app_state_mock,
        train_data_loader=llm_data_loader_mock,
        evaluation_data_loaders=[],
        checkpoint_saving=checkpoint_saving_mock,
        training_log_interval_in_steps=1,
        checkpointing_interval_in_steps=1,
        evaluation_interval_in_steps=1,
    )
    app_state_mock.model_parts[0].assert_has_calls([call(b.samples) for b in batches])
    app_state_mock.lr_scheduler.step.assert_called()


def test_dummy_lr_scheduler(optimizer_with_param_groups_mock: MagicMock):
    # we test that the optimizer step function reduces the lr by 0.01 for each param group.
    # we also test that the scheduler step function does not change the lr.

    scheduler = DummyLRScheduler(optimizer=optimizer_with_param_groups_mock)
    assert scheduler.get_lr() == [0.1, 0.2, 0.3]
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert scheduler.get_last_lr() == [0.1, 0.2, 0.3]

    optimizer_with_param_groups_mock.step()
    assert np.allclose(scheduler.get_lr(), [0.09, 0.19, 0.29], atol=1e-6)
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert scheduler.get_last_lr() == [0.1, 0.2, 0.3]

    scheduler.step()
    assert np.allclose(scheduler.get_lr(), [0.09, 0.19, 0.29], atol=1e-6)
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert np.allclose(scheduler.get_last_lr(), [0.09, 0.19, 0.29], atol=1e-6)

    optimizer_with_param_groups_mock.step()
    assert np.allclose(scheduler.get_lr(), [0.08, 0.18, 0.28], atol=1e-6)
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert np.allclose(scheduler.get_last_lr(), [0.09, 0.19, 0.29], atol=1e-6)

    scheduler.step()
    assert np.allclose(scheduler.get_lr(), [0.08, 0.18, 0.28], atol=1e-6)
    assert scheduler._get_closed_form_lr() == [0.1, 0.2, 0.3]
    assert np.allclose(scheduler.get_last_lr(), [0.08, 0.18, 0.28], atol=1e-6)


def test_linear_warmup_cosine_annealing_lr_scheduler():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = LRSchedulerFactory.get_linear_warmup_cosine_annealing_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=2,
        total_steps=6,
        initial_lr=0.1,
        final_lr=0.2,
        max_lr=1.0,
    )

    learning_rates = [scheduler.get_last_lr()[0]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        learning_rates.append(scheduler.get_last_lr()[0])

    assert learning_rates[0] < learning_rates[1] < learning_rates[2]
    assert np.isclose(learning_rates[2], 1.0, atol=1e-6)
    assert learning_rates[2] > learning_rates[3] > learning_rates[4] > learning_rates[5] > learning_rates[6]
    assert np.isclose(learning_rates[6], 0.2, atol=1e-6)
