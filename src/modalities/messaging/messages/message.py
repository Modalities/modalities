from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


class MessageTypes(Enum):
    STEP_STATE = "STEP_STATE"
    BATCH_PROGRESS_UPDATE = "PROGRESS_UPDATE"
    MODEL_STATE = "MODEL_STATE"
    EVALUATION_RESULT = "EVALUATION_RESULT"


T = TypeVar("T")


@dataclass
class Message(Generic[T]):
    """An object representing a message."""

    message_type: Enum
    payload: T
    global_rank: int = 0
    local_rank: int = 0
