from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, InferenceResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.loops.training.gradient_clipping.gradient_clipper import GradientClipperIF
from modalities.loss_functions import Loss
from modalities.messaging.evaluation.processors.standard_step_state_processor import TrackablesKeys
from modalities.messaging.messages.message import MessageTypes
from modalities.messaging.messages.payloads import BatchProgressUpdate, ExperimentStatus, StepState
from modalities.messaging.publishers.publisher import MessagePublisher
from modalities.models.model import model_predict_batch
from modalities.util import TimeRecorder


class TrainingLoop:
    def __init__(
        self,
        local_rank: int,
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        step_state_publisher: MessagePublisher[StepState],
        gradient_acc_steps: int,
        gradient_clipper: GradientClipperIF,
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.step_state_publisher = step_state_publisher
        self.gradient_acc_steps = gradient_acc_steps
        self.gradient_clipper = gradient_clipper

    def _train_batch(
        self,
        batch: DatasetBatch,
        model: FSDP,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        train_step_id: int,
        data_loader: LLMDataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor, InferenceResultBatch]:
        result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch)
        (loss / self.gradient_acc_steps).backward()

        if (train_step_id + 1) % self.gradient_acc_steps == 0 or (train_step_id + 1) == len(data_loader):
            gradient_norm_score = self.gradient_clipper.clip_gradients().sum()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            return loss, gradient_norm_score, result_batch
        else:
            return loss, None, result_batch

    def train(
        self,
        model: nn.Module,
        train_loader: LLMDataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        evaluation_callback: Callable[[int], None],
        checkpointing_callback: Callable[[int], None],
    ):
        model.train()
        # cumulated_losses = self._reset_tracked_losses()

        torch.device(self.local_rank if torch.cuda.is_available() else "cpu")

        # batch loop
        batch: DatasetBatch
        # TODO: why do we need a barrier here?
        dist.barrier()
        forward_backward_time_recorder = TimeRecorder()
        forward_backward_time_recorder.start()

        for batch_id, batch in enumerate(train_loader):
            # Because we might resume training, we add the starting batch id of the data loader
            train_step_id = batch_id + train_loader.fast_forward_batch_id

            self._publish_batch_progress_update(
                batch_progress_publisher=self.batch_progress_publisher,
                train_step_id=train_step_id,
                num_steps=len(train_loader),
                dataloader_tag=train_loader.dataloader_tag,
            )

            # Train single batch
            batch_loss, gradient_norm_score, result_batch = self._train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fun=loss_fun,
                train_step_id=train_step_id,
                data_loader=train_loader,
            )
            forward_backward_time_recorder.stop()

            trackable_values = {
                TrackablesKeys.NUM_SAMPLES: len(batch),
                TrackablesKeys.FORWARD_BACKWARD_TIME: forward_backward_time_recorder.delta_t,
                TrackablesKeys.NUM_STEPS: 1,
                TrackablesKeys.CUMM_BATCH_LOSS: batch_loss.item(),
                TrackablesKeys.LAST_BATCH_LOSS: batch_loss.item(),
                TrackablesKeys.LAST_SCHEDULER_LR: scheduler.get_last_lr()[0],
            }
            if gradient_norm_score is not None:
                trackable_values[TrackablesKeys.LAST_BATCH_GRADIENT_NORM] = gradient_norm_score.item()

            train_step_state = StepState(
                trackable_values=trackable_values,
                inference_result_batch=result_batch,
                meta_information=StepState.MetaInformation(
                    step_id=train_step_id,
                    num_steps=len(train_loader),
                    dataloader_tag=train_loader.dataloader_tag,
                    loss_fun_tag=loss_fun.tag,
                    experiment_status=ExperimentStatus.TRAIN,
                ),
            )

            # send the train step state to the broker
            self.step_state_publisher.publish_message(payload=train_step_state, message_type=MessageTypes.STEP_STATE)

            evaluation_callback(train_step_id=train_step_id)
            checkpointing_callback(train_step_id=train_step_id)

            # we start the time recoder here again to also capture the time spend loading
            # via the dataloader.
            forward_backward_time_recorder.reset()
            forward_backward_time_recorder.start()
            model.train()

    @staticmethod
    def _publish_batch_progress_update(
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        train_step_id: int,
        num_steps: int,
        dataloader_tag: str,
    ):
        payload = BatchProgressUpdate(
            step_id=train_step_id,
            num_steps=num_steps,
            experiment_status=ExperimentStatus.TRAIN,
            dataloader_tag=dataloader_tag,
        )
        batch_progress_publisher.publish_message(payload=payload, message_type=MessageTypes.BATCH_PROGRESS_UPDATE)
