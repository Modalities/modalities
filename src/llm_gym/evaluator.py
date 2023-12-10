from typing import Callable, Dict, List
import torch
from llm_gym.batch import DatasetBatch, InferenceResultBatch, EvaluationResultBatch
from llm_gym.dataloader.dataloader import LLMDataLoader
import torch.distributed as dist
from llm_gym.batch import DatasetBatch, EvaluationResultBatch, InferenceResultBatch
from llm_gym.fsdp.reducer import Reducer
from llm_gym.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from llm_gym.logging_broker.publisher import MessagePublisher
from llm_gym.models.model import NNModel, model_predict_batch


class Evaluator:
    def __init__(
        self,
        local_rank: int,
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
    ) -> None:
        self.local_rank = local_rank
        self.batch_progress_publisher = batch_progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher

    def evaluate_batch(
        self,
        batch: DatasetBatch,
        model: NNModel,
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
    ):
        with torch.no_grad():
            result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch)
        return loss

    def evaluate(
        self,
        model: NNModel,
        data_loaders: List[LLMDataLoader],
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
        train_batch_id: int,
    ) -> Dict[str, EvaluationResultBatch]:
        result_dict: Dict[str, EvaluationResultBatch] = {}
        model.eval()
        for data_loader in data_loaders:
            cummulated_loss = torch.zeros(3).to(self.local_rank)
            Evaluator._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                train_batch_id=train_batch_id,
                dataset_batch_id=-1,
                dataloader_tag=data_loader.dataloader_tag,
            )
            # for _, (batch_id, batch) in zip(range(1000), enumerate(data_loader)):
            for batch_id, batch in enumerate(data_loader):
                batch_loss = self.evaluate_batch(
                    batch=batch,
                    model=model,
                    loss_fun=loss_fun,
                )

                cummulated_loss[0] += batch_loss.item()  # sum up batch loss
                cummulated_loss[1] += len(batch)

                Evaluator._publish_progress(
                    batch_progress_publisher=self.batch_progress_publisher,
                    train_batch_id=train_batch_id,
                    dataset_batch_id=batch_id,
                    dataloader_tag=data_loader.dataloader_tag,
                )
            # TODO: insert reducer from outside so Evaluator is independent of FSDP
            total_loss = Reducer.reduce(
                tensor=cummulated_loss,
                operation=dist.ReduceOp.SUM,
                post_processing_fun=lambda t: t[0] / t[1],
            )

            evaluation_result = EvaluationResultBatch(
                losses={loss_fun.tag: total_loss},
                dataloader_tag=data_loader.dataloader_tag,
                train_batch_id=train_batch_id,
            )
            Evaluator._publish_evaluation_result(
                evaluation_result_publisher=self.evaluation_result_publisher,
                evaluation_result=evaluation_result,
            )
            result_dict[data_loader.dataloader_tag] = evaluation_result
        # Evaluator._publish_progress(
        #     batch_progress_publisher=self.batch_progress_publisher,
        #     train_batch_id=train_batch_id + 1,
        #     dataset_batch_id=0,
        #     dataloader_tag=data_loader.dataloader_tag,
        # )
        return result_dict

    @staticmethod
    def _publish_progress(
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        train_batch_id: int,
        dataset_batch_id: int,
        dataloader_tag: str,
    ):
        payload = BatchProgressUpdate(
            train_batch_id=train_batch_id,
            dataset_batch_id=dataset_batch_id,
            experiment_status=ExperimentStatus.EVALUATION,
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
