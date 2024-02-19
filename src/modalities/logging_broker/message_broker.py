from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

from modalities.logging_broker.messages import Message, MessageTypes
from modalities.logging_broker.subscriber import MessageSubscriberIF


class MessageBrokerIF(ABC):
    """Interface for message broker objects."""

    @abstractmethod
    def add_subscriber(self, subscription: MessageTypes, subscriber: MessageSubscriberIF):
        raise NotImplementedError

    @abstractmethod
    def distribute_message(self, message: Message):
        raise NotImplementedError


class MessageBroker(MessageBrokerIF):
    """The MessageBroker sends notifications to its subscribers."""

    def __init__(self) -> None:
        self.subscriptions: Dict[MessageTypes, List[MessageSubscriberIF]] = defaultdict(list)

    def add_subscriber(self, subscription: MessageTypes, subscriber: MessageSubscriberIF):
        """Adds a single subscriber."""
        self.subscriptions[subscription].append(subscriber)

    def distribute_message(self, message: Message):
        """Distributes message to all subscribers."""
        message_type = message.message_type
        for subscriber in self.subscriptions[message_type]:
            subscriber.consume_message(message=message)
