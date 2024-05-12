from enum import Enum
from typing import Dict, List, Optional

import torch.distributed as dist
from pydantic import BaseModel

from modalities.messaging.broker.message_broker import MessageBrokerIF
from modalities.messaging.evaluation.processors.batch_progress_update_processors import BatchProgressUpdateProcessor
from modalities.messaging.evaluation.processors.processors import GlobalProcessorIF, LocalProcessorIF
from modalities.messaging.evaluation.processors.standard_step_state_processor import (
    StandardGlobalStepStateProcessor,
    StandardLocalStepStateProcessor,
)
from modalities.messaging.evaluation.states import IntervalState
from modalities.messaging.messages.message import Message, MessageTypes
from modalities.messaging.messages.payloads import EvaluationResult, ExperimentStatus, StepState
from modalities.messaging.subscribers.subscriber import MessageSubscriberIF


class DistributedEvaluation(MessageSubscriberIF):
    def __init__(
        self,
        message_broker: MessageBrokerIF,
        training_log_interval_in_steps: int,
        message_type_subscriptions: List[MessageTypes],
        local_processors: Optional[Dict[Enum, List[LocalProcessorIF]]] = None,
        global_processors: Optional[List[GlobalProcessorIF]] = None,
    ) -> None:
        self.message_broker = message_broker
        self.training_log_interval_in_steps = training_log_interval_in_steps
        self.interval_state: IntervalState = None

        # subscribe to the relevant messages
        for message_type in message_type_subscriptions:
            self.message_broker.add_subscriber(subscription=message_type, subscriber=self)

        # specify the local message processors
        self.local_processors: Dict[Enum, List[LocalProcessorIF]] = {
            MessageTypes.STEP_STATE: [StandardLocalStepStateProcessor()],
            MessageTypes.MODEL_STATE: [],
            MessageTypes.BATCH_PROGRESS_UPDATE: [BatchProgressUpdateProcessor()],
        }
        if local_processors is not None:
            for message_type, processors in local_processors.items():
                self.local_processors[message_type] += processors

        # specify the global processor that work on the aggregated global state,
        # after reducing the local states across the ranks
        world_size = dist.get_world_size()
        self.global_processors: List[GlobalProcessorIF] = [StandardGlobalStepStateProcessor(world_size=world_size)]
        if global_processors is not None:
            self.global_processors += global_processors

    def _publish_eval_result_message(self, payload: EvaluationResult):
        message = Message(message_type=MessageTypes.EVALUATION_RESULT, payload=payload)
        self.message_broker.distribute_message(message)

    def consume_message(self, message: Message):
        self._process_local(message)

        if message.message_type == MessageTypes.STEP_STATE:
            step_state: StepState = message.payload
            step_id = step_state.meta_information.step_id
            is_log_step = (
                (step_id + 1) % self.training_log_interval_in_steps == 0
                and self.interval_state.meta_information.experiment_status == ExperimentStatus.TRAIN
            )
            is_last_step = (step_id + 1) == step_state.meta_information.num_steps
            if is_log_step or is_last_step:
                self.interval_state.reduce_values_across_ranks()
                eval_result = self._process_global()
                self._publish_eval_result_message(payload=eval_result)
                self._reset_train_step_state()

    def _process_local(self, message: Message):
        for local_processor in self.local_processors[message.message_type]:
            local_processor.process(payload=message.payload, current_local_step_state=self.interval_state)

    def _process_global(self) -> EvaluationResult:
        eval_result = EvaluationResult(
            dataloader_tag=self.interval_state.meta_information.dataloader_tag,
            train_step_id=self.interval_state.meta_information.step_id,
            experiment_status=self.interval_state.meta_information.experiment_status,
        )
        for global_processor in self.global_processors:
            global_processor.process(interval_state=self.interval_state, eval_result=eval_result)
        return eval_result

    def _reset_train_step_state(self):
        self.interval_state = None


class DistributedEvaluationConfig(BaseModel):
    message_broker: MessageBrokerIF
    training_log_interval_in_steps: int
    message_type_subscriptions: List[MessageTypes]
    local_processors: Optional[Dict[Enum, List[LocalProcessorIF]]] = None
    global_processors: Optional[List[GlobalProcessorIF]] = None
