from dataclasses import dataclass, field

from modalities.training.training_progress import TrainingProgress


@dataclass
class CheckpointingInstruction:
    """
    Represents a checkpointing instruction (i.e., saving and deleting).

    Attributes:
        save_current (bool): Indicates whether to save the current checkpoint.
        checkpoints_to_delete (list[TrainingProgress]): List of checkpoint IDs to delete.
    """

    save_current: bool = False
    checkpoints_to_delete: list[TrainingProgress] = field(default_factory=list)
