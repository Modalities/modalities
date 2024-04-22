from enum import Enum
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, EvaluationResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import model_predict_batch
from modalities.running_env.fsdp.reducer import Reducer
from modalities.util import Aggregator, TimeRecorder


class ThroughputAggregationKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"


class Trainer:
    def __init__(
        self,
        local_rank: int,
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        gradient_acc_steps: int,
        gradient_clipper: Callable[[nn.Module], torch.Tensor],
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self.gradient_acc_steps = gradient_acc_steps
        self.gradient_clipper = gradient_clipper

    def _train_batch(
        self,
        batch: DatasetBatch,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        train_step_id: int,
        data_loader: LLMDataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch) / self.gradient_acc_steps
        loss.backward()
        gradient_norm_score = self.gradient_clipper(model)

        if (train_step_id + 1) % self.gradient_acc_steps == 0 or (train_step_id + 1) == len(data_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        return loss, gradient_norm_score

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
        cumulated_loss_and_gradient_norm = self._reset_loss_and_gradient_norm()

        thoughput_aggregator = Aggregator[ThroughputAggregationKeys]()

        device = torch.device(self.local_rank if torch.cuda.is_available() else "cpu")

        # batch loop
        batch: DatasetBatch
        # TODO: why do we need a barrier here?
        dist.barrier()
        forward_backward_time_recorder = TimeRecorder()
        forward_backward_time_recorder.start()
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
            cumulated_loss_and_gradient_norm[0] += batch_loss.item()
            cumulated_loss_and_gradient_norm[1] += gradient_norm_score.item()
            # This works, because we always drop the last batch in case it has less samples than the batch size
            cumulated_loss_and_gradient_norm[-1] += 1  # number of local batches
            batch_length_tensor = torch.tensor(len(batch)).to(device)
            thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)

            self._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                train_step_id=train_step_id,
                dataloader_tag=train_loader.dataloader_tag,
            )

            # Check, if model should be evaluated
            if (train_step_id + 1) % global_training_log_interval_in_steps == 0:
                forward_backward_time = torch.tensor(forward_backward_time_recorder.delta_t).to(device)
                forward_backward_time_recorder.reset()

                thoughput_aggregator.add_value(
                    key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_time
                )
                synced_num_samples = thoughput_aggregator.get_all_reduced_value(ThroughputAggregationKeys.NUM_SAMPLES)
                synced_forward_backward_time = thoughput_aggregator.get_all_reduced_value(
                    ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
                )
                synced_num_samples_per_second = synced_num_samples / synced_forward_backward_time
                # TODO: insert reducer from outside so Trainer is independent of FSDP
                cumulated_loss_and_gradient_norm[2] = batch_loss.item()
                cumulated_loss_and_gradient_norm[3] = gradient_norm_score.item()

                reduced_loss_and_gradient_norm = Reducer.reduce(
                    tensor=cumulated_loss_and_gradient_norm,
                    operation=dist.ReduceOp.SUM,
                    # divide the first two elements by the last one
                    # i.e., summed batch loss / (num batches * world size)
                    # and summed gradient norm/ (num batches * world size).
                    # keep the other elements as is
                    post_processing_fun=lambda t: torch.cat((t[:2] / t[-1], t[2:-1] / dist.get_world_size())),
                )

                train_loss_avg, train_gradient_norm_avg, train_loss_last_batch, train_gradient_norm_last_batch = (
                    reduced_loss_and_gradient_norm[0],
                    reduced_loss_and_gradient_norm[1],
                    reduced_loss_and_gradient_norm[2],
                    reduced_loss_and_gradient_norm[3],
                )

                training_metrics = EvaluationResultBatch(
                    losses={
                        f"{loss_fun.tag} interval average": train_loss_avg,
                        f"{loss_fun.tag} last batch": train_loss_last_batch,
                    },
                    metrics={
                        "grad_norm_avg": train_gradient_norm_avg,
                        "grad_norm_last_batch": train_gradient_norm_last_batch,
                    },
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
                thoughput_aggregator.remove_keys()

                model.train()
                cumulated_loss_and_gradient_norm = self._reset_loss_and_gradient_norm()

            evaluation_callback(train_step_id=train_step_id)
            checkpointing_callback(train_step_id=train_step_id)
            # we start the time recoder here again to also capture the time spend loading
            # via the dataloader.
            forward_backward_time_recorder.start()

    def _reset_loss_and_gradient_norm(self):
        # TODO: we should handle the device assignment more centrally.
        cumulated_loss_and_gradient_norm = torch.zeros(5)
        if torch.cuda.is_available():
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to(torch.device(self.local_rank))
        else:
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to("cpu")
        return cumulated_loss_and_gradient_norm

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
