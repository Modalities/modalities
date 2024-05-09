from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


class MessageTypes(Enum):
    FORWARD_BACKWARD_PASS_STATE = "FORWARD_BACKWARD_PASS_STATE"

    HIGH_LEVEL_PROGRESS_UPDATE = "HIGH_LEVEL_PROGRESS_UPDATE"
    BATCH_PROGRESS_UPDATE = "PROGRESS_UPDATE"
    ERROR_MESSAGE = "ERROR_MESSAGE"
    EVALUATION_RESULT = "EVALUATION_RESULT"
    MODEL_STATE = "MODEL_STATE"


T = TypeVar("T")


@dataclass
class Message(Generic[T]):
    """An object representing a message."""

    message_type: MessageTypes
    payload: T
    global_rank: int = 0
    local_rank: int = 0
