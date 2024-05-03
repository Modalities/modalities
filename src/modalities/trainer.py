from enum import Enum
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, EvaluationResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import model_predict_batch
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF
from modalities.util import Aggregator, TimeRecorder


class AggregationKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"

    NUM_STEPS = "NUM_STEPS"
    CUMM_LOSS = "CUMM_LOSS"
    LAST_BATCH_LOSS = "LAST_BATCH_LOSS"


class Trainer:
    def __init__(
        self,
        local_rank: int,
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        gradient_acc_steps: int,
        gradient_clipper: GradientClipperIF,
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch)
        (loss / self.gradient_acc_steps).backward()

        if (train_step_id + 1) % self.gradient_acc_steps == 0 or (train_step_id + 1) == len(data_loader):
            gradient_norm_score = self.gradient_clipper.clip_gradients().sum()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            return loss, gradient_norm_score
        else:
            return loss, None

    def train(
        self,
        model: nn.Module,
        train_loader: LLMDataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        global_training_log_interval_in_steps: int,
        evaluation_callback: Callable[[int], None],
        checkpointing_callback: Callable[[int], None],
    ):
        model.train()
        # cumulated_losses = self._reset_tracked_losses()

        score_aggregator = Aggregator[AggregationKeys]()

        torch.device(self.local_rank if torch.cuda.is_available() else "cpu")

        # batch loop
        batch: DatasetBatch
        # TODO: why do we need a barrier here?
        dist.barrier()
        forward_backward_time_recorder = TimeRecorder()
        forward_backward_time_recorder.start()
        gradient_norm_scores = []
        for batch_id, batch in enumerate(train_loader):
            # Because we might resume training, we add the starting batch id of the data loader
            train_step_id = batch_id + train_loader.fast_forward_batch_id
            # Train single batch
            batch_loss, gradient_norm_score = self._train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fun=loss_fun,
                train_step_id=train_step_id,
                data_loader=train_loader,
            )
            forward_backward_time_recorder.stop()
            # Save the batch loss
            score_aggregator.add_value(key=AggregationKeys.CUMM_LOSS, value=batch_loss.item())
            # This works, because we always drop the last batch in case it has less samples than the batch size
            score_aggregator.add_value(key=AggregationKeys.NUM_STEPS, value=1)

            # gradient norm is already synced across all ranks
            if gradient_norm_score is not None:
                gradient_norm_scores.append(gradient_norm_score.item())

            score_aggregator.add_value(key=AggregationKeys.NUM_SAMPLES, value=len(batch))

            self._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                train_step_id=train_step_id,
                dataloader_tag=train_loader.dataloader_tag,
            )

            # Check, if model should be evaluated
            if (train_step_id + 1) % global_training_log_interval_in_steps == 0:
                # add the loss for the LAST batch
                score_aggregator.add_value(key=AggregationKeys.LAST_BATCH_LOSS, value=batch_loss.item())
                score_aggregator.add_value(
                    key=AggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_time_recorder.delta_t
                )

                forward_backward_time_recorder.reset()

                # reduce the scores with the respective reduction operation
                sum_reduced_scores = score_aggregator.get_all_reduced_values(
                    keys=[
                        AggregationKeys.NUM_SAMPLES,
                        AggregationKeys.NUM_STEPS,
                        AggregationKeys.CUMM_LOSS,
                        AggregationKeys.LAST_BATCH_LOSS,
                    ],
                    reduce_operation=dist.ReduceOp.SUM,
                )

                max_reduced_scores = score_aggregator.get_all_reduced_values(
                    keys=[AggregationKeys.FORWARD_BACKWARD_TIME], reduce_operation=dist.ReduceOp.MAX
                )

                # calculate the metric scores for logging
                synced_num_samples_per_second = (
                    sum_reduced_scores[AggregationKeys.NUM_SAMPLES]
                    / max_reduced_scores[AggregationKeys.FORWARD_BACKWARD_TIME]
                )
                train_loss_avg = (
                    sum_reduced_scores[AggregationKeys.CUMM_LOSS] / sum_reduced_scores[AggregationKeys.NUM_STEPS]
                )
                train_loss_last_batch = sum_reduced_scores[AggregationKeys.LAST_BATCH_LOSS] / dist.get_world_size()

                losses = {
                    f"{loss_fun.tag} average": train_loss_avg,
                    f"{loss_fun.tag} last step": train_loss_last_batch,
                }
                if len(gradient_norm_scores) > 0:
                    metrics = {
                        "grad_norm_avg": torch.mean(torch.Tensor(gradient_norm_scores)),
                        "grad_norm_last_batch": gradient_norm_scores[-1],
                    }
                    gradient_norm_scores = []
                else:
                    metrics = {}

                training_metrics = EvaluationResultBatch(
                    losses=losses,
                    metrics=metrics,
                    # TODO: hardcoded metric key
                    throughput_metrics={
                        "training_synced_num_samples_per_second": synced_num_samples_per_second,
                        "lr_mean": torch.tensor(scheduler.get_last_lr()).mean(),
                        "lr_min": torch.tensor(scheduler.get_last_lr()).min(),
                        "lr_max": torch.tensor(scheduler.get_last_lr()).max(),
                        "lr_first": torch.tensor(scheduler.get_last_lr())[0],
                    },
                    dataloader_tag=train_loader.dataloader_tag,
                    train_step_id=train_step_id,
                )
                self._publish_evaluation_result(
                    evaluation_result_publisher=self.evaluation_result_publisher,
                    evaluation_result=training_metrics,
                )
                score_aggregator.remove_keys()

                model.train()

            evaluation_callback(train_step_id=train_step_id)
            checkpointing_callback(train_step_id=train_step_id)
            # we start the time recoder here again to also capture the time spend loading
            # via the dataloader.
            forward_backward_time_recorder.start()

    @staticmethod
    def _publish_progress(
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        train_step_id: int,
        dataloader_tag: str,
    ):
        payload = BatchProgressUpdate(
            step_id=train_step_id,
            experiment_status=ExperimentStatus.TRAIN,
            dataloader_tag=dataloader_tag,
        )
        batch_progress_publisher.publish_message(payload=payload, message_type=MessageTypes.BATCH_PROGRESS_UPDATE)

    @staticmethod
    def _publish_evaluation_result(
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        evaluation_result: EvaluationResultBatch,
    ):
        evaluation_result_publisher.publish_message(
            payload=evaluation_result, message_type=MessageTypes.EVALUATION_RESULT
        )
