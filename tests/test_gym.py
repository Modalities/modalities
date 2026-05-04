from unittest.mock import call

from pytest import MonkeyPatch

from modalities.checkpointing.checkpoint_saving import CheckpointSaving
from modalities.checkpointing.stateful.app_state import AppState
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.evaluator import Evaluator
from modalities.gym import Gym
from modalities.loss_functions import Loss
from modalities.trainer import Trainer
from tests.utility import configure_dataloader_mock


def test_run_cpu_only(
    set_env_cpu,
    monkeypatch: MonkeyPatch,
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
        training_log_interval_in_steps=1,
        checkpointing_interval_in_steps=1,
        evaluation_interval_in_steps=1,
        train_data_loader=llm_data_loader_mock,
        evaluation_data_loaders=[],
        checkpoint_saving=checkpoint_saving_mock,
    )
    app_state_mock.model_parts[0].assert_has_calls([call(b.samples) for b in batches])
    app_state_mock.optimizer.step.assert_called()
