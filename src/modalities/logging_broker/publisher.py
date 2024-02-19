from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from modalities.logging_broker.message_broker import Message, MessageBroker
from modalities.logging_broker.messages import MessageTypes

T = TypeVar("T")


class MessagePublisherIF(ABC, Generic[T]):
    @abstractmethod
    def publish_message(self, payload: T, message_type: MessageTypes):
        raise NotImplementedError


class MessagePublisher(MessagePublisherIF[T]):
    """The MessagePublisher sends messages through a message broker."""

    def __init__(
        self,
        message_broker: MessageBroker,
        global_rank: int,
        local_rank: int,
    ):
        self.message_broker = message_broker
        self.global_rank = global_rank
        self.local_rank = local_rank

    def publish_message(self, payload: T, message_type: MessageTypes):
        """Publish a message through the message broker."""
        message = Message[T](
            message_type=message_type,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            payload=payload,
        )
        self.message_broker.distribute_message(message)
