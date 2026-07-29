"""Tests for DCP checkpoint deletion.

A distributed checkpoint is a *directory* of per-rank shard files. Deleting one therefore has to be
recursive; ``Path.rmdir()`` only removes empty directories and always raised
``OSError("Directory not empty")``, which silently broke every rotating checkpoint strategy that was
combined with the DCP execution.
"""

import json
from pathlib import Path

import pytest

from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import DCPCheckpointSaving
from modalities.exceptions import CheckpointingError
from modalities.training.training_progress import TrainingProgress

EXPERIMENT_ID = "test_experiment"


def _training_progress() -> TrainingProgress:
    return TrainingProgress(
        num_seen_steps_current_run=8,
        num_seen_tokens_current_run=64,
        num_target_steps=16,
        num_target_tokens=128,
    )


def _make_saving(checkpoint_path: Path, global_rank: int = 0) -> DCPCheckpointSaving:
    return DCPCheckpointSaving(
        checkpoint_path=checkpoint_path,
        experiment_id=EXPERIMENT_ID,
        global_rank=global_rank,
    )


def _create_checkpoint_folder(saving: DCPCheckpointSaving, training_progress: TrainingProgress) -> Path:
    """Creates a folder shaped like a real DCP checkpoint: a directory containing shard files."""
    folder = saving._get_checkpointing_folder_path(
        experiment_id=EXPERIMENT_ID,
        num_seen_steps=training_progress.num_seen_steps_total,
        num_seen_tokens=training_progress.num_seen_tokens_total,
        num_target_steps=training_progress.num_target_steps,
        num_target_tokens=training_progress.num_target_tokens,
    )
    folder.mkdir(parents=True)
    # This is what makes rmdir() fail: DCP writes one shard file per rank plus metadata.
    (folder / ".metadata").write_bytes(b"metadata")
    for rank in range(4):
        (folder / f"__{rank}_0.distcp").write_bytes(b"shard")
    (folder / "nested").mkdir()
    (folder / "nested" / "extra.bin").write_bytes(b"nested")
    return folder


def test_deletes_a_non_empty_checkpoint_directory(tmp_path):
    # The regression: a DCP checkpoint folder is never empty, so deletion must be recursive.
    saving = _make_saving(tmp_path)
    training_progress = _training_progress()
    folder = _create_checkpoint_folder(saving, training_progress)
    assert any(folder.iterdir()), "test fixture must produce a non-empty directory"

    saving._delete_checkpoint(training_progress=training_progress)

    assert not folder.exists()
    # Only the checkpoint folder goes away; the surrounding checkpoint directory stays.
    assert tmp_path.exists()


def test_deletion_leaves_sibling_checkpoints_untouched(tmp_path):
    saving = _make_saving(tmp_path)
    to_delete = _training_progress()
    keep = TrainingProgress(
        num_seen_steps_current_run=16,
        num_seen_tokens_current_run=128,
        num_target_steps=16,
        num_target_tokens=128,
    )
    folder_to_delete = _create_checkpoint_folder(saving, to_delete)
    folder_to_keep = _create_checkpoint_folder(saving, keep)
    info_file = tmp_path / "last_checkpoint_info.json"
    info_file.write_text(json.dumps({"checkpoint_folder_path": str(folder_to_keep)}))

    saving._delete_checkpoint(training_progress=to_delete)

    assert not folder_to_delete.exists()
    assert folder_to_keep.exists()
    assert sorted(p.name for p in folder_to_keep.iterdir()) == [
        ".metadata",
        "__0_0.distcp",
        "__1_0.distcp",
        "__2_0.distcp",
        "__3_0.distcp",
        "nested",
    ]
    assert info_file.exists()


def test_non_zero_rank_does_not_delete(tmp_path):
    # Only rank 0 removes checkpoints; every other rank must be a no-op.
    saving = _make_saving(tmp_path, global_rank=1)
    training_progress = _training_progress()
    folder = _create_checkpoint_folder(saving, training_progress)

    saving._delete_checkpoint(training_progress=training_progress)

    assert folder.exists()


def test_missing_checkpoint_raises(tmp_path):
    saving = _make_saving(tmp_path)
    with pytest.raises(CheckpointingError, match="does not exist"):
        saving._delete_checkpoint(training_progress=_training_progress())


def test_refuses_to_delete_a_file(tmp_path):
    saving = _make_saving(tmp_path)
    training_progress = _training_progress()
    folder = saving._get_checkpointing_folder_path(
        experiment_id=EXPERIMENT_ID,
        num_seen_steps=training_progress.num_seen_steps_total,
        num_seen_tokens=training_progress.num_seen_tokens_total,
        num_target_steps=training_progress.num_target_steps,
        num_target_tokens=training_progress.num_target_tokens,
    )
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.write_bytes(b"not a directory")

    with pytest.raises(CheckpointingError, match="is not a directory"):
        saving._delete_checkpoint(training_progress=training_progress)
    assert folder.exists()


def test_refuses_to_delete_outside_the_configured_checkpoint_path(tmp_path):
    # A recursive delete must not be able to escape the configured checkpoint directory, even if the
    # experiment id were ever to contain path separators.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.bin").write_bytes(b"do not delete")

    checkpoint_path = tmp_path / "checkpoints"
    checkpoint_path.mkdir()
    saving = DCPCheckpointSaving(
        checkpoint_path=checkpoint_path,
        experiment_id="../outside/escaped",
        global_rank=0,
    )
    training_progress = _training_progress()
    folder = saving._get_checkpointing_folder_path(
        experiment_id="../outside/escaped",
        num_seen_steps=training_progress.num_seen_steps_total,
        num_seen_tokens=training_progress.num_seen_tokens_total,
        num_target_steps=training_progress.num_target_steps,
        num_target_tokens=training_progress.num_target_tokens,
    )
    folder.mkdir(parents=True)
    (folder / "shard.distcp").write_bytes(b"shard")

    with pytest.raises(CheckpointingError, match="Refusing to delete"):
        saving._delete_checkpoint(training_progress=training_progress)
    assert (outside / "precious.bin").exists()
    assert folder.exists()


def test_rotation_end_to_end_with_the_save_k_most_recent_strategy(tmp_path):
    # The failure this fixes only showed up through the strategy: it asks the execution to delete the
    # checkpoint that just fell out of the k-most-recent window.
    from modalities.checkpointing.checkpoint_saving_strategies import SaveKMostRecentCheckpointsStrategy

    saving = _make_saving(tmp_path)
    strategy = SaveKMostRecentCheckpointsStrategy(k=1)

    first = _training_progress()
    second = TrainingProgress(
        num_seen_steps_current_run=16,
        num_seen_tokens_current_run=128,
        num_target_steps=16,
        num_target_tokens=128,
    )
    first_folder = _create_checkpoint_folder(saving, first)
    _create_checkpoint_folder(saving, second)

    instruction = strategy.get_checkpoint_instruction(training_progress=first)
    assert instruction.save_current
    instruction = strategy.get_checkpoint_instruction(training_progress=second)
    assert instruction.checkpoints_to_delete, "strategy should ask for the older checkpoint to go"

    # Deleting the evicted checkpoint must now succeed rather than raise OSError.
    saving._delete_checkpoint(training_progress=first)
    assert not first_folder.exists()
