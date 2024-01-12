from dataclasses import dataclass, field
from typing import List


@dataclass
class CheckpointingInstruction:
    """
    Instruction to save and delete checkpoints.
    """

    save_current: bool = False
    checkpoints_to_delete: List[int] = field(default_factory=list)
