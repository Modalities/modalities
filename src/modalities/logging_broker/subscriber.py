from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from modalities.logging_broker.messages import Message

T = TypeVar("T")


class MessageSubscriberIF(ABC, Generic[T]):
    """Interface for message subscribers."""

    @abstractmethod
    def consume_message(self, message: Message[T]):
        raise NotImplementedError
    
