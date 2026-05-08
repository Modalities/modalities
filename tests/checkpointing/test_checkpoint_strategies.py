import dataclasses

import pytest

from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction
from modalities.checkpointing.checkpoint_saving_strategies import (
    KeepEveryKStepsAndMMostRecentCheckpointingStrategy,
    SaveKMostRecentCheckpointsStrategy,
)
from modalities.training.training_progress import TrainingProgress


@pytest.mark.parametrize(
    "k, saved_instances, checkpoints_to_delete, save_current",
    [
        # k value is 2. New checkpoint is created and the last one (in the example: [2]) is deleted.
        (2, [TrainingProgress(2, 2, 20, 20), TrainingProgress(1, 1, 20, 20)], [TrainingProgress(1, 1, 20, 20)], True),
        # k value is 0. No deletion of checkpoints.
        (0, [], [], False),
        # k value is 2, but there are currently only one checkpoint. Hence, no deletion.
        (2, [TrainingProgress(1, 1, 20, 20)], [], True),
        # k value is -1, therefore we want to keep all checkpoints without any deletion
        (
            -1,
            [TrainingProgress(3, 3, 20, 20), TrainingProgress(2, 2, 20, 20), TrainingProgress(1, 1, 20, 20)],
            [],
            True,
        ),
    ],
)
def test_checkpoint_strategy_k(
    k: int, saved_instances: list[TrainingProgress], checkpoints_to_delete: list[int], save_current: bool
) -> None:
    num_seen_steps_current_run = 10
    training_progress = TrainingProgress(
        num_seen_steps_current_run=num_seen_steps_current_run,
        num_seen_tokens_current_run=10,
        num_target_steps=20,
        num_target_tokens=40,
    )
    checkpoint_strategy = SaveKMostRecentCheckpointsStrategy(k=k)
    checkpoint_strategy.saved_step_checkpoints = saved_instances
    checkpoint_instruction = checkpoint_strategy.get_checkpoint_instruction(training_progress=training_progress)

    assert checkpoint_instruction.checkpoints_to_delete == checkpoints_to_delete
    assert checkpoint_instruction.save_current == save_current

    # make sure that modifying the training progress externally does not affect saved_step_checkpoints
    if k != 0 and save_current:
        training_progress.num_seen_steps_current_run = 100
        assert checkpoint_strategy.saved_step_checkpoints[0].num_seen_steps_current_run == num_seen_steps_current_run


@pytest.mark.parametrize(
    "k, num_recent_checkpoints_to_keep, num_steps",
    [
        (3, 2, 11),
        (2, 1, 10),
        (4, 3, 15),
    ],
)
def test_keep_every_k_steps_keeps_every_k_steps(k: int, num_recent_checkpoints_to_keep: int, num_steps: int) -> None:
    checkpoint_strategy = KeepEveryKStepsAndMMostRecentCheckpointingStrategy(
        k=k, num_recent_checkpoints_to_keep=num_recent_checkpoints_to_keep
    )
    training_progress = TrainingProgress(
        num_seen_steps_current_run=0,
        num_seen_tokens_current_run=0,
        num_target_steps=20,
        num_target_tokens=40,
    )

    # Simulate training progress and checkpointing
    simulator = _CheckpointSavingSimulator()
    for step in range(1, num_steps + 1):
        training_progress.num_seen_steps_current_run = step
        checkpoint_instruction = checkpoint_strategy.get_checkpoint_instruction(training_progress=training_progress)
        simulator.simulate_training_step(training_progress, checkpoint_instruction)

    for ckpt in simulator.saved_checkpoints:
        # Check that only checkpoints that are divisible by k or the most recent ones are kept.
        last_checkpoints = set(range(num_steps - num_recent_checkpoints_to_keep + 1, num_steps + 1))
        assert ckpt.num_seen_steps_current_run % k == 0 or ckpt.num_seen_steps_current_run in last_checkpoints


def test_keep_every_k_steps_checkpointing_strategy_invalid_arguments() -> None:
    with pytest.raises(AssertionError):
        KeepEveryKStepsAndMMostRecentCheckpointingStrategy(k=0, num_recent_checkpoints_to_keep=1)
    with pytest.raises(AssertionError):
        KeepEveryKStepsAndMMostRecentCheckpointingStrategy(k=-1, num_recent_checkpoints_to_keep=1)
    with pytest.raises(AssertionError):
        KeepEveryKStepsAndMMostRecentCheckpointingStrategy(k=2, num_recent_checkpoints_to_keep=0)
    with pytest.raises(AssertionError):
        KeepEveryKStepsAndMMostRecentCheckpointingStrategy(k=2, num_recent_checkpoints_to_keep=-1)


class _CheckpointSavingSimulator:
    def __init__(self):
        self.saved_checkpoints: list[TrainingProgress] = []

    def simulate_training_step(
        self, training_progress: TrainingProgress, ckpt_instruction: CheckpointingInstruction
    ) -> None:
        if ckpt_instruction.save_current:
            self.saved_checkpoints.append(dataclasses.replace(training_progress))
        for checkpoint_to_delete in ckpt_instruction.checkpoints_to_delete:
            self.saved_checkpoints = [cp for cp in self.saved_checkpoints if cp != checkpoint_to_delete]
