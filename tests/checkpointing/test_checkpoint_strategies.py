import pytest

from modalities.checkpointing.checkpoint_saving_strategies import SaveKMostRecentCheckpointsStrategy
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
