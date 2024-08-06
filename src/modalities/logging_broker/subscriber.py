from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

from modalities.logging_broker.messages import Message

T = TypeVar("T")


class MessageSubscriberIF(ABC, Generic[T]):
    """Interface for message subscribers."""

    @abstractmethod
    def consume_message(self, message: Message[T]):
        raise NotImplementedError

    @abstractmethod
    def consume_dict(self, mesasge_dict: Dict[str, Any]):
        raise NotImplementedError
