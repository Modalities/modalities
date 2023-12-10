from typing import List

import pytest

from src.llm_gym.checkpointing.checkpointing import CheckpointingInstruction
from src.llm_gym.checkpointing.checkpointing_strategies import SaveKMostRecentCheckpointsStrategy


@pytest.mark.parametrize(
    "k, saved_batch_id_checkpoints, global_train_batch_id, checkpoints_to_delete",
    [
        (2, [1, 2], 100, [2]),
        # Test case 1: k value is 2. New checkpoint is created and the last one (in the example: [2]) is deleted.
        (0, [1, 2], 100, []),  # Test case 2: k value is 0. No deletion of checkpoints.
        (2, [1], 100, []),
        # Test case 1: k value is 2, but there are currently only one checkpoint. Hence, no deletion.
    ])
def test_checkpoint_strategy_k(
        k: int,
        saved_batch_id_checkpoints: List[int],
        global_train_batch_id: int,
        checkpoints_to_delete: List[int]
) -> None:
    checkpoint_strategy = SaveKMostRecentCheckpointsStrategy(k=k)
    checkpoint_strategy.saved_batch_id_checkpoints = saved_batch_id_checkpoints
    checkpoint_instruction = checkpoint_strategy.get_checkpoint_instruction(
        global_train_batch_id=global_train_batch_id,
        num_batches=5
    )

    assert checkpoint_instruction.checkpoints_to_delete == checkpoints_to_delete
