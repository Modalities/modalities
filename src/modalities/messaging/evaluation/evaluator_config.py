from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

from modalities.config.pydanctic_if_types import (
    PydanticGlobalProcessorIFType,
    PydanticLocalProcessorIFType,
    PydanticMessageBrokerIFType,
)
from modalities.messaging.messages.message import MessageTypes


class DistributedEvaluationConfig(BaseModel):
    message_broker: PydanticMessageBrokerIFType
    training_log_interval_in_steps: int
    message_type_subscriptions: List[MessageTypes]
    local_processors: Optional[Dict[Enum, List[PydanticLocalProcessorIFType]]] = None
    global_processors: Optional[List[PydanticGlobalProcessorIFType]] = None
