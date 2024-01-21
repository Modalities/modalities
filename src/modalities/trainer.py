from enum import Enum
from typing import Callable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

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
        gradient_acc_step: int,
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self.gradient_acc_step = gradient_acc_step

    def _train_batch(
        self,
        batch: DatasetBatch,
        model: NNModel,
        optimizer: Optimizer,
        loss_fun: Loss,
        batch_id: int,
        data_loader: LLMDataLoader,
    ) -> torch.Tensor:
        result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch) / self.gradient_acc_step
        loss.backward()

        if (batch_id + 1) % self.gradient_acc_step == 0 or (batch_id + 1) == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()
        return loss

    def train(
        self,
        model: NNModel,
        train_loader: LLMDataLoader,
        optimizer,
        loss_fun: Loss,
        callback_interval_in_batches: int,
        epoch_done_callback: Callable[[int], None],
        local_sample_id_to_global_sample_id: Callable[[int], int],
    ):
        model.train()
        cummulated_loss = self._reset_loss()
        thoughput_aggregator = Aggregator[ThroughputAggregationKeys]()

        # batch loop
        batch: DatasetBatch
        # TODO: why do we need a barrier here?
        dist.barrier()
        forward_backward_time_recorder = TimeRecorder()
        forward_backward_time_recorder.start()
        for batch_id, batch in enumerate(train_loader):
            # Because we might resume training, we add the starting batch id of the data loader
            local_train_batch_id = batch_id + train_loader.fast_forward_batch_id
            # train single batch
            batch_loss = self._train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                loss_fun=loss_fun,
                batch_id=batch_id,
                data_loader=train_loader,
            )
            forward_backward_time_recorder.stop()
            # save the batch loss
            cummulated_loss[0] += batch_loss.item()
            cummulated_loss[1] += len(batch)
            batch_length_tensor = torch.tensor(len(batch)).to(torch.device(self.local_rank))
            thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)
            self._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                local_batch_id=local_train_batch_id,
                batch_size=train_loader.sampler_batch_size,
                dataloader_tag=train_loader.dataloader_tag,
                local_sample_id_to_global_sample_id=local_sample_id_to_global_sample_id,
            )

            # Check, if model should be evaluated
            if (local_train_batch_id + 1) % callback_interval_in_batches == 0:
                if local_train_batch_id > 0:
                    foward_backward_time = torch.tensor(forward_backward_time_recorder.delta_t).to(
                        torch.device(self.local_rank)
                    )
                    forward_backward_time_recorder.reset()

                    thoughput_aggregator.add_value(
                        key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=foward_backward_time
                    )
                    synced_num_samples = thoughput_aggregator.get_all_reduced_value(
                        ThroughputAggregationKeys.NUM_SAMPLES
                    )
                    synced_foward_backward_time = thoughput_aggregator.get_all_reduced_value(
                        ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
                    )
                    synced_num_samples_per_second = synced_num_samples / synced_foward_backward_time
                    # TODO: insert reducer from outside so Trainer is independent of FSDP
                    train_loss = Reducer.reduce(
                        tensor=cummulated_loss,
                        operation=dist.ReduceOp.SUM,
                        post_processing_fun=lambda t: t[0] / t[1],
                    )
                    local_train_sample_id = Trainer._get_local_sample_id(
                        batch_id=local_train_batch_id, batch_size=train_loader.sampler_batch_size
                    )

                    global_train_sample_id = local_sample_id_to_global_sample_id(local_train_sample_id)

                    evaluation_result = EvaluationResultBatch(
                        losses={loss_fun.tag: train_loss},
                        # TODO: hardcoded metric key
                        throughput_metrics={"training_synced_num_samples_per_second": synced_num_samples_per_second},
                        dataloader_tag=train_loader.dataloader_tag,
                        global_train_sample_id=global_train_sample_id,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=self.evaluation_result_publisher,
                        evaluation_result=evaluation_result,
                    )
                    #epoch_done_callback(local_train_sample_id=local_train_sample_id)
                    thoughput_aggregator.remove_keys()
                    #epoch_done_callback(local_train_sample_id=local_train_sample_id)
                    model.train()

                # TODO early stopping
                cummulated_loss = self._reset_loss()
            # we start the time recoder here again to also capture the time spend loading
            # via the dataloader.
            forward_backward_time_recorder.start()



    def _reset_loss(self):
        # TODO: we should handle the device assignment more centrally.
        cummulated_loss = torch.zeros(2)
        if torch.cuda.is_available():
            cummulated_loss = cummulated_loss.to(torch.device(self.local_rank))
        else:
            cummulated_loss = cummulated_loss.to("cpu")
        return cummulated_loss

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
