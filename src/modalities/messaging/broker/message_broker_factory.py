from typing import Dict, List

from modalities.messaging.broker.message_broker import MessageBroker
from modalities.messaging.messages.message import MessageTypes
from modalities.messaging.subscribers.subscriber import MessageSubscriberIF


class MessageBrokerFactory:
    @staticmethod
    def get_message_broker(subscriptions: Dict[MessageTypes, List[MessageSubscriberIF]]) -> MessageBroker:
        message_broker = MessageBroker()

        for subscription, subscribers in subscriptions.items():
            for subscriber in subscribers:
                message_broker.add_subscriber(subscription, subscriber)
        return message_broker
