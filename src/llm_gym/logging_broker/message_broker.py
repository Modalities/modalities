from abc import ABC, abstractmethod
from collections import defaultdict
from llm_gym.logging_broker.messages import Message, MessageTypes
from llm_gym.logging_broker.subscriber import MessageSubscriberIF
from typing import Dict, List


class MessageBrokerIF(ABC):
    @abstractmethod
    def add_subscriber(self, subscription: MessageTypes, subscriber: MessageSubscriberIF):
        raise NotImplementedError

    @abstractmethod
    def distribute_message(self, message: Message):
        raise NotImplementedError


class MessageBroker(MessageBrokerIF):
    def __init__(self) -> None:
        self.subscriptions: Dict[MessageTypes, List[MessageSubscriberIF]] = defaultdict(list)

    def add_subscriber(self, subscription: MessageTypes, subscriber: MessageSubscriberIF):
        self.subscriptions[subscription].append(subscriber)

    def distribute_message(self, message: Message):
        message_type = message.message_type
        for subscriber in self.subscriptions[message_type]:
            subscriber.consume_message(message=message)
