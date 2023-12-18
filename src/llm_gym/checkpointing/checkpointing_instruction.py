from typing import List
from dataclasses import dataclass, field


@dataclass
class CheckpointingInstruction:
    """
    Instruction to save and delete checkpoints.
    """

    save_current: bool = False
    checkpoints_to_delete: List[int] = field(default_factory=list)