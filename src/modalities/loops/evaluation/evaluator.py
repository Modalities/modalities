from typing import Annotated, Callable, Dict, List, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from modalities.batch import DatasetBatch, InferenceResultBatch
from modalities.config.pydanctic_if_types import (
    PydanticBatchProgressUpdatePublisherIFType,
    PydanticStepStatePublisherIFType,
)
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.messaging.evaluation.processors.standard_step_state_processor import TrackablesKeys
from modalities.messaging.messages.message import MessageTypes
from modalities.messaging.messages.payloads import BatchProgressUpdate, EvaluationResult, ExperimentStatus, StepState
from modalities.messaging.publishers.publisher import MessagePublisher
from modalities.models.model import model_predict_batch
from modalities.util import TimeRecorder


class Evaluator:
    def __init__(
        self,
        local_rank: int,
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        step_state_publisher: MessagePublisher[StepState],
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.step_state_publisher = step_state_publisher

    def evaluate_batch(
        self,
        batch: DatasetBatch,
        model: nn.Module,
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
    ) -> Union[torch.Tensor, InferenceResultBatch]:
        with torch.no_grad():
            result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch)
        return loss, result_batch

    def evaluate(
        self,
        model: nn.Module,
        data_loaders: List[LLMDataLoader],
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
        train_step_id: int,
    ) -> Dict[str, EvaluationResult]:
        model.eval()
        for data_loader in data_loaders:
            Evaluator._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                eval_step_id=0,  # Reset progress bar
                num_steps=len(data_loader),
                dataloader_tag=data_loader.dataloader_tag,
            )
            forward_backward_time_recorder = TimeRecorder()
            forward_backward_time_recorder.start()
            for batch_id, batch in enumerate(data_loader):
                Evaluator._publish_progress(
                    batch_progress_publisher=self.batch_progress_publisher,
                    eval_step_id=batch_id,
                    num_steps=len(data_loader),
                    dataloader_tag=data_loader.dataloader_tag,
                )

                batch_loss, result_batch = self.evaluate_batch(
                    batch=batch,
                    model=model,
                    loss_fun=loss_fun,
                )
                forward_backward_time_recorder.stop()

                trackable_values = {
                    TrackablesKeys.NUM_SAMPLES: len(batch),
                    TrackablesKeys.FORWARD_BACKWARD_TIME: forward_backward_time_recorder.delta_t,
                    TrackablesKeys.NUM_STEPS: 1,
                    TrackablesKeys.CUMM_BATCH_LOSS: batch_loss.item(),
                    TrackablesKeys.LAST_BATCH_LOSS: batch_loss.item(),
                }

                eval_step_state = StepState(
                    trackable_values=trackable_values,
                    inference_result_batch=result_batch,
                    meta_information=StepState.MetaInformation(
                        step_id=train_step_id,
                        num_steps=len(data_loader),
                        dataloader_tag=data_loader.dataloader_tag,
                        loss_fun_tag=loss_fun.tag,
                        experiment_status=ExperimentStatus.EVALUATION,
                    ),
                )
                # send the train step state to the broker
                self.step_state_publisher.publish_message(payload=eval_step_state, message_type=MessageTypes.STEP_STATE)
                # we start the time recoder here again to also capture the time spend loading
                # via the dataloader.
                forward_backward_time_recorder.reset()
                forward_backward_time_recorder.start()

    @staticmethod
    def _publish_progress(
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        eval_step_id: int,
        num_steps: int,
        dataloader_tag: str,
    ):
        payload = BatchProgressUpdate(
            step_id=eval_step_id,
            num_steps=num_steps,
            experiment_status=ExperimentStatus.EVALUATION,
            dataloader_tag=dataloader_tag,
        )
        batch_progress_publisher.publish_message(payload=payload, message_type=MessageTypes.BATCH_PROGRESS_UPDATE)


class EvaluatorConfig(BaseModel):
    local_rank: Annotated[int, Field(strict=True, ge=1)]
    batch_progress_publisher: PydanticBatchProgressUpdatePublisherIFType
    step_state_publisher: PydanticStepStatePublisherIFType
