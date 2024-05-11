from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, InferenceResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.loss_functions import Loss
from modalities.messaging.messages.message import MessageTypes
from modalities.messaging.messages.payloads import BatchProgressUpdate, ExperimentStatus, TrainStepState
from modalities.messaging.publishers.publisher import MessagePublisher
from modalities.models.model import model_predict_batch
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF
from modalities.util import TimeRecorder


class Trainer:
    def __init__(
        self,
        local_rank: int,
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        forward_backward_pass_publisher: MessagePublisher[TrainStepState],
        gradient_acc_steps: int,
        gradient_clipper: GradientClipperIF,
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.forward_backward_pass_publisher = forward_backward_pass_publisher
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

            train_step_state = TrainStepState(
                trackables=TrainStepState.Trackables(
                    loss=batch_loss.item(),
                    gradient_norm_score=gradient_norm_score.item() if gradient_norm_score is not None else None,
                    num_samples=len(batch),
                    forward_backward_time=forward_backward_time_recorder.delta_t,
                ),
                inference_result_batch=result_batch,
                meta_information=TrainStepState.MetaInformation(
                    step_id=train_step_id,
                    dataloader_tag=train_loader.dataloader_tag,
                    loss_fun_tag=loss_fun.tag,
                    experiment_status=ExperimentStatus.TRAIN,
                ),
            )

            # send the train step state to the broker
            self.forward_backward_pass_publisher.publish_message(
                payload=train_step_state, message_type=MessageTypes.FORWARD_BACKWARD_PASS_STATE
            )

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
