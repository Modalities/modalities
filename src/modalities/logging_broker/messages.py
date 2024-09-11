from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


class MessageTypes(Enum):
    HIGH_LEVEL_PROGRESS_UPDATE = "HIGH_LEVEL_PROGRESS_UPDATE"
    BATCH_PROGRESS_UPDATE = "PROGRESS_UPDATE"
    ERROR_MESSAGE = "ERROR_MESSAGE"
    EVALUATION_RESULT = "EVALUATION_RESULT"


T = TypeVar("T")


@dataclass
class Message(Generic[T]):
    """An object representing a message."""

    message_type: MessageTypes
    payload: T
    global_rank: int = 0
    local_rank: int = 0


class ExperimentStatus(Enum):
    TRAIN = "TRAIN"
    EVALUATION = "EVALUATION"


@dataclass
class ProgressUpdate:
    """Object holding the state of the current batch / step computation progress."""

    num_steps_done: int
    # Note: in case of ExperimentState.TRAIN, dataset_batch_id=global_train_batch_id
    experiment_status: ExperimentStatus
    dataloader_tag: str
