from typing import Callable
from llm_gym.fsdp.reducer import Reducer
from llm_gym.logging_broker.messages import (
    BatchProgressUpdate,
    ExperimentStatus,
    MessageTypes,
)
from llm_gym.logging_broker.publisher import MessagePublisher
from llm_gym.loss_functions import Loss
from llm_gym.models.model import NNModel, model_predict_batch
import torch.distributed as dist
import torch
from llm_gym.batch import DatasetBatch, EvaluationResultBatch
from torch.optim import Optimizer
from llm_gym.dataloader.dataloader import LLMDataLoader


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
        eval_interval_in_batches: int,
        num_batches_per_rank: int,
        epoch_done_callback: Callable[[int], None],
    ):
        model.train()
        cummulated_loss = torch.zeros(2).to(self.local_rank)

        # batch loop
        batch: DatasetBatch
        for train_batch_id, batch in zip(range(num_batches_per_rank), train_loader):
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
                train_batch_id=train_batch_id,
                dataset_tag=train_loader.dataset_tag,
            )

            # Check, if model should be evaluated
            if (train_batch_id + 1) % eval_interval_in_batches == 0:
                if train_batch_id > 0:
                    # TODO: insert reducer from outside so Trainer is independent of FSDP
                    train_loss = Reducer.reduce(
                        tensor=cummulated_loss,
                        operation=dist.ReduceOp.SUM,
                        post_processing_fun=lambda t: t[0] / t[1],
                    )
                    evaluation_result = EvaluationResultBatch(
                        losses={loss_fun.tag: train_loss},
                        dataset_tag=train_loader.dataset_tag,
                        train_batch_id=train_batch_id,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=self.evaluation_result_publisher,
                        evaluation_result=evaluation_result,
                    )
                    epoch_done_callback(
                        train_batch_id=train_batch_id,
                    )
                    model.train()

                # TODO early stopping

                # reset loss
                # TODO in the future we should outsource this functionality
                cummulated_loss = torch.zeros(2).to(self.local_rank)

    @staticmethod
    def _publish_progress(
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        train_batch_id: int,
        dataset_tag: str,
    ):
        payload = BatchProgressUpdate(
            train_batch_id=train_batch_id,
            dataset_batch_id=train_batch_id,
            experiment_status=ExperimentStatus.TRAIN,
            dataset_tag=dataset_tag,
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
