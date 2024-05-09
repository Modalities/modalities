from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from modalities.batch import EvaluationResultBatch
from modalities.messaging.broker.message_broker import MessageBrokerIF
from modalities.messaging.evaluation.aggregator import DistributedAggregator
from modalities.messaging.messages.message import Message, MessageTypes
from modalities.messaging.messages.payloads import BatchProgressUpdate, EvalStepState, ExperimentStatus, TrainStepState
from modalities.messaging.publishers.publisher import MessagePublisherIF
from modalities.messaging.subscribers.subscriber import MessageSubscriberIF


class AggregationKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"
    NUM_STEPS = "NUM_STEPS"
    CUMM_BATCH_LOSS = "CUMM_BATCH_LOSS"
    LAST_BATCH_LOSS = "LAST_BATCH_LOSS"


class LocalSingleStepStateProcessor:
    def __init__(self):
        pass


class LocalMultiStepStateProcessor:
    def __init__(self):
        pass


class GlobalMultiStepStateProcessor:
    def __init__(self):
        pass


class LocalStepStateFilter:
    def __init__(self):
        pass


class RankReduceOperations(dist.ReduceOp):
    NONE = "NONE"


class LocalReduceOperations(Enum):
    def SUM(last: Tuple, current) -> float | int:
        return last + current

    def MAX(last: Tuple, current) -> float | int:
        return np.max(last, current)

    def REPLACE(_: Tuple, current) -> float | int:
        return current


@dataclass
class Trackable:
    key: Enum
    value: float | int | torch.Tensor
    rank_reduce_op: RankReduceOperations
    local_reduce_op = LocalReduceOperations


@dataclass
class CurrentLocalStepState:
    class TrackableCollection:
        def __init__(self):
            self.state: Dict[Enum, float | int] = {}

        def set(self, trackable: Trackable):
            if trackable.key in self.state:
                local_reduce_op = trackable.local_reduce_op.value
                self.state[trackable.key] = local_reduce_op([self.state[trackable.key], trackable.value])
            else:
                self.state[trackable.key] = trackable.value

    @dataclass
    class MetaInformation:
        step_id: int
        num_steps: int
        dataloader_tag: str
        experiment_status: ExperimentStatus

    trackables: TrackableCollection
    meta_information: MetaInformation


# pipeline
# pass payload through local step state processor
# publish the trackables calculated by the local step state processors
# if the step is the last step, sync the trackables with reduce_op != NONE across all ranks
# pass payload through global step state processor
# publish the trackables calculated by the global step state processors


class DistributedEvaluator(MessageSubscriberIF, MessagePublisherIF[EvaluationResultBatch]):
    def __init__(self, message_broker: MessageBrokerIF, training_log_interval_in_steps: int) -> None:
        self.message_broker = message_broker
        self.message_broker.add_subscriber(subscription=TrainStepState, subscriber=self)
        self.message_broker.add_subscriber(subscription=BatchProgressUpdate, subscriber=self)

        self.training_log_interval_in_steps = training_log_interval_in_steps
        self.current_batch_progress_update: BatchProgressUpdate = None

        self.train_aggregator = DistributedAggregator[AggregationKeys]()
        self.train_step_state_history: List[TrainStepState] = []

    def publish_message(self, payload: EvaluationResultBatch, message_type: MessageTypes):
        message = Message(message_type=message_type, payload=payload)
        self.message_broker.distribute_message(message)

    def consume_message(self, message: Message):
        raise NotImplementedError

    def _consume_batch_progress_update(self, batch_progress_update: BatchProgressUpdate):
        # TODO check if the step id is always communicated BEFORE the step is executed!
        self.current_batch_progress_update = batch_progress_update

    def _consume_eval_step_state(self, eval_step_state: EvalStepState):
        self.train_step_state_history.append(eval_step_state)
        eval_step_id = eval_step_state.meta_information.step_id
        if (eval_step_id + 1) == eval_step_state.meta_information.num_steps:
            self._process_eval_step_states()

    def _consume_train_step_state(self, train_step_state: TrainStepState):
        self.train_step_state_history.append(train_step_state)
        train_step_id = train_step_state.meta_information.step_id
        # Check, if model should be evaluated
        if (train_step_id + 1) % self.training_log_interval_in_steps == 0:
            self._process_train_step_states()

    def _process_train_step_states(self):
        for train_step_state in self.train_step_state_history:
            # Save the batch loss
            self.train_aggregator.add_value(key=AggregationKeys.CUMM_BATCH_LOSS, value=train_step_state.trackables.loss)
            self.train_aggregator.add_value(key=AggregationKeys.NUM_STEPS, value=1)
            self.train_aggregator.add_value(
                key=AggregationKeys.NUM_SAMPLES, value=train_step_state.trackables.num_samples
            )
            self.train_aggregator.add_value(
                key=AggregationKeys.FORWARD_BACKWARD_TIME, value=train_step_state.trackables.forward_backward_time
            )

        # add the loss for the LAST batch
        last_train_step_state = self.train_step_state_history[-1]
        self.train_aggregator.add_value(
            key=AggregationKeys.LAST_BATCH_LOSS, value=last_train_step_state.trackables.loss
        )

        sum_reduced_scores, max_reduced_scores = self._get_reduced_scores()

        if dist.get_rank() == 0:
            result_batch = self._get_evaluation_result_batch(
                last_train_step_state=last_train_step_state,
                sum_reduced_scores=sum_reduced_scores,
                max_reduced_scores=max_reduced_scores,
            )

        # share the result batch with the subscribers, e.g., for logging
        self.publish_message(payload=result_batch, message_type=MessageTypes.EVALUATION_RESULT)

        self._reset_train_step_state()

    def _get_reduced_scores(self) -> Tuple[Dict, Dict]:
        # reduce the scores with the respective reduction operation
        sum_reduced_scores = self.train_aggregator.get_all_reduced_values(
            keys=[
                AggregationKeys.NUM_SAMPLES,
                AggregationKeys.NUM_STEPS,
                AggregationKeys.CUMM_BATCH_LOSS,
                AggregationKeys.LAST_BATCH_LOSS,
            ],
            reduce_operation=dist.ReduceOp.SUM,
        )

        max_reduced_scores = self.train_aggregator.get_all_reduced_values(
            keys=[AggregationKeys.FORWARD_BACKWARD_TIME], reduce_operation=dist.ReduceOp.MAX
        )
        return sum_reduced_scores, max_reduced_scores

    def _get_evaluation_result_batch(
        self,
        last_train_step_state: TrainStepState,
        sum_reduced_scores: Dict[AggregationKeys, torch.Tensor],
        max_reduced_scores: Dict[AggregationKeys, torch.Tensor],
    ) -> EvaluationResultBatch:
        # throughput
        synced_num_samples_per_second = (
            sum_reduced_scores[AggregationKeys.NUM_SAMPLES] / max_reduced_scores[AggregationKeys.FORWARD_BACKWARD_TIME]
        )

        # losses
        train_loss_avg = (
            sum_reduced_scores[AggregationKeys.CUMM_BATCH_LOSS] / sum_reduced_scores[AggregationKeys.NUM_STEPS]
        )

        train_loss_last_batch = sum_reduced_scores[AggregationKeys.LAST_BATCH_LOSS] / dist.get_world_size()

        losses = {
            f"{last_train_step_state.meta_information.loss_fun_tag} average": train_loss_avg,
            f"{last_train_step_state.meta_information.loss_fun_tag} last step": train_loss_last_batch,
        }

        # gradient norm
        gradient_norm_scores = [
            payload.trackables.gradient_norm_score
            for payload in self.train_step_state_history
            if payload.trackables.gradient_norm_score is not None
        ]

        if len(gradient_norm_scores) > 0:
            metrics = {
                "grad_norm_avg": np.mean(gradient_norm_scores),
                "grad_norm_last_batch": gradient_norm_scores[-1],
            }
        else:
            metrics = {}

        result_batch = EvaluationResultBatch(
            losses=losses,
            metrics=metrics,
            # TODO: hardcoded metric key
            meta_metrics={
                "training_synced_num_samples_per_second": synced_num_samples_per_second,
                "scheduler_lr_first": last_train_step_state.trackables.scheduler_lr_first,
            },
            dataloader_tag=last_train_step_state.meta_information.dataloader_tag,
            train_step_id=last_train_step_state.meta_information.step_id,
        )
        return result_batch

    def _reset_train_step_state(self):
        self.train_step_state_history = []
        self.train_aggregator.remove_keys()
