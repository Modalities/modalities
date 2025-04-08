from typing import Any

from pydantic import BaseModel

from modalities.batch import EvaluationResultBatch
from modalities.logging_broker.messages import Message
from modalities.logging_broker.subscriber import MessageSubscriberIF


class SaveAllResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def __init__(self):
        self.message_list: list[Message[EvaluationResultBatch]] = []

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        self.message_list.append(message)

    def consume_dict(self, mesasge_dict: dict[str, Any]):
        pass


class SaveAllResultSubscriberConfig(BaseModel):
    pass
