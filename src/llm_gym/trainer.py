from typing import Callable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from llm_gym.batch import DatasetBatch, EvaluationResultBatch
from llm_gym.dataloader.dataloader import LLMDataLoader
from llm_gym.fsdp.reducer import Reducer
from llm_gym.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from llm_gym.logging_broker.publisher import MessagePublisher
from llm_gym.loss_functions import Loss
from llm_gym.models.model import NNModel, model_predict_batch


class Trainer:
    def __init__(
        self,
        local_rank: int,
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher

    def _train_batch(
        self,
        batch: DatasetBatch,
        model: NNModel,
        optimizer: Optimizer,
        loss_fun: Loss,
    ) -> torch.Tensor:
        optimizer.zero_grad()
        result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch)
        loss.backward()
        optimizer.step()
        return loss

    def train(
        self,
        model: NNModel,
        train_loader: LLMDataLoader,
        optimizer,
        loss_fun: Loss,
        callback_interval_in_batches: int,
        local_num_train_samples: int,
        epoch_done_callback: Callable[[int], None],
        local_sample_id_to_global_sample_id: Callable[[int], int],
    ):
        model.train()
        cummulated_loss = self._reset_loss()

        # batch loop
        batch: DatasetBatch
        for local_train_batch_id, batch in zip(range(local_num_train_samples), train_loader):
            # train single batch
            batch_loss = self._train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                loss_fun=loss_fun,
            )

            # save the batch loss
            cummulated_loss[0] += batch_loss.item()
            cummulated_loss[1] += len(batch)

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
                        dataloader_tag=train_loader.dataloader_tag,
                        global_train_sample_id=global_train_sample_id,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=self.evaluation_result_publisher,
                        evaluation_result=evaluation_result,
                    )
                    epoch_done_callback(
                        local_train_sample_id=Trainer._get_local_sample_id(
                            batch_id=local_train_batch_id, batch_size=train_loader.sampler_batch_size
                        )
                    )
                    model.train()

                # TODO early stopping
                cummulated_loss = self._reset_loss()

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
