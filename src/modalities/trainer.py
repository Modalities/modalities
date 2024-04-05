from enum import Enum
from typing import Callable, Tuple

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, EvaluationResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import NNModel, model_predict_batch
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
        gradient_clipper: Callable[[NNModel], torch.Tensor],
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self.gradient_acc_steps = gradient_acc_steps
        self.gradient_clipper = gradient_clipper

    def _train_batch(
        self,
        batch: DatasetBatch,
        model: NNModel,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        batch_id: int,
        data_loader: LLMDataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch) / self.gradient_acc_steps
        loss.backward()
        gradient_norm_score = self.gradient_clipper(model)

        if (batch_id + 1) % self.gradient_acc_steps == 0 or (batch_id + 1) == len(data_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        return loss, gradient_norm_score

    def train(
        self,
        model: NNModel,
        train_loader: LLMDataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        callback_interval_in_batches: int,
        # TODO: remove
        epoch_done_callback: Callable[[int], None],
        local_sample_id_to_global_sample_id: Callable[[int], int],
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
            local_train_batch_id = batch_id + train_loader.fast_forward_batch_id
            # Train single batch
            batch_loss, gradient_norm_score = self._train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fun=loss_fun,
                batch_id=batch_id,
                data_loader=train_loader,
            )
            forward_backward_time_recorder.stop()
            # Save the batch loss
            cumulated_loss_and_gradient_norm[0] += batch_loss.item()
            cumulated_loss_and_gradient_norm[1] += gradient_norm_score.item()
            cumulated_loss_and_gradient_norm[-1] += len(batch)
            batch_length_tensor = torch.tensor(len(batch)).to(device)
            thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)
            self._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                local_batch_id=local_train_batch_id,
                batch_size=train_loader.batch_size,
                dataloader_tag=train_loader.dataloader_tag,
                local_sample_id_to_global_sample_id=local_sample_id_to_global_sample_id,
            )

            # Check, if model should be evaluated
            if (local_train_batch_id + 1) % callback_interval_in_batches == 0:
                if local_train_batch_id > 0:
                    forward_backward_time = torch.tensor(forward_backward_time_recorder.delta_t).to(device)
                    forward_backward_time_recorder.reset()

                    thoughput_aggregator.add_value(
                        key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_time
                    )
                    synced_num_samples = thoughput_aggregator.get_all_reduced_value(
                        ThroughputAggregationKeys.NUM_SAMPLES
                    )
                    synced_forward_backward_time = thoughput_aggregator.get_all_reduced_value(
                        ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
                    )
                    synced_num_samples_per_second = synced_num_samples / synced_forward_backward_time
                    # TODO: insert reducer from outside so Trainer is independent of FSDP
                    reduced_loss_and_gradient_norm = Reducer.reduce(
                        tensor=cumulated_loss_and_gradient_norm,
                        operation=dist.ReduceOp.SUM,
                        post_processing_fun=lambda t: t[0:-1] / t[-1],
                    )
                    train_loss, train_gradient_norm = (
                        reduced_loss_and_gradient_norm[0],
                        reduced_loss_and_gradient_norm[1],
                    )
                    local_train_sample_id = Trainer._get_local_sample_id(
                        batch_id=local_train_batch_id, batch_size=train_loader.batch_size
                    )

                    global_train_sample_id = local_sample_id_to_global_sample_id(local_train_sample_id)

                    evaluation_result = EvaluationResultBatch(
                        losses={loss_fun.tag: train_loss},
                        metrics={"grad_norm": train_gradient_norm},
                        # TODO: hardcoded metric key
                        throughput_metrics={
                            "training_synced_num_samples_per_second": synced_num_samples_per_second,
                            "lr_mean": torch.tensor(scheduler.get_last_lr()).mean(),
                            "lr_min": torch.tensor(scheduler.get_last_lr()).min(),
                            "lr_max": torch.tensor(scheduler.get_last_lr()).max(),
                            "lr_first": torch.tensor(scheduler.get_last_lr())[0],
                        },
                        dataloader_tag=train_loader.dataloader_tag,
                        global_train_sample_id=global_train_sample_id,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=self.evaluation_result_publisher,
                        evaluation_result=evaluation_result,
                    )
                    thoughput_aggregator.remove_keys()
                    epoch_done_callback(local_train_sample_id=local_train_sample_id)
                    model.train()

                # TODO early stopping
                cumulated_loss_and_gradient_norm = self._reset_loss_and_gradient_norm()
            # we start the time recoder here again to also capture the time spend loading
            # via the dataloader.
            forward_backward_time_recorder.start()

    def _reset_loss_and_gradient_norm(self):
        # TODO: we should handle the device assignment more centrally.
        cumulated_loss_and_gradient_norm = torch.zeros(3)
        if torch.cuda.is_available():
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to(torch.device(self.local_rank))
        else:
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to("cpu")
        return cumulated_loss_and_gradient_norm

    @staticmethod
    def _publish_progress(
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        local_batch_id: int,
        batch_size: int,
        dataloader_tag: str,
        local_sample_id_to_global_sample_id: Callable[[int], int],
    ):
        local_train_sample_id = Trainer._get_local_sample_id(batch_id=local_batch_id, batch_size=batch_size)
        global_train_sample_id = local_sample_id_to_global_sample_id(local_train_sample_id)

        payload = BatchProgressUpdate(
            global_train_sample_id=global_train_sample_id,
            global_dataset_sample_id=global_train_sample_id,
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

    @staticmethod
    def _get_local_sample_id(batch_id: int, batch_size: int) -> int:
        return (batch_id + 1) * batch_size - 1
