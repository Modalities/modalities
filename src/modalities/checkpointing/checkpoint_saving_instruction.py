from dataclasses import dataclass, field
from typing import List


@dataclass
class CheckpointingInstruction:
    """
    Represents a checkpointing instruction (i.e., saving and deleting).

    Attributes:
        save_current (bool): Indicates whether to save the current checkpoint.
        checkpoints_to_delete (List[int]): List of checkpoint IDs to delete.
    """

    save_current: bool = False
    checkpoints_to_delete: List[int] = field(default_factory=list)
