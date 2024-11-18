from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from modalities.batch import DatasetBatch, EvaluationResultBatch, InferenceResultBatch, ResultItem
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import ExperimentStatus, MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.models.model import model_predict_batch
from modalities.running_env.fsdp.reducer import Reducer
from modalities.trainer import ThroughputAggregationKeys
from modalities.util import Aggregator, TimeRecorder


class Evaluator:
    """Evaluator class which is responsible for evaluating the model on a set of datasets"""

    def __init__(
        self,
        progress_publisher: MessagePublisher[ProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
    ) -> None:
        """Initializes the Evaluator class.

        Args:
            progress_publisher (MessagePublisher[ProgressUpdate]): Publisher for progress updates
            evaluation_result_publisher (MessagePublisher[EvaluationResultBatch]): Publisher for evaluation results
        """
        self.progress_publisher = progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher

    def evaluate_batch(
        self,
        batch: DatasetBatch,
        model: nn.Module,
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate a single batch by forwarding it through the model and calculating the loss.

        Args:
            batch (DatasetBatch): The batch to evaluate
            model (nn.Module): The model to evaluate
            loss_fun (Callable[[InferenceResultBatch], torch.Tensor]): The loss function to calculate the loss

        Returns:
            torch.Tensor: The loss of the batch
        """
        with torch.no_grad():
            result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch)
        return loss

    def evaluate(
        self,
        model: nn.Module,
        data_loaders: list[LLMDataLoader],
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
        num_train_steps_done: int,
    ) -> dict[str, EvaluationResultBatch]:
        """Evaluate the model on a set of datasets.

        Args:
            model (nn.Module): The model to evaluate
            data_loaders (list[LLMDataLoader]): List of dataloaders to evaluate the model on
            loss_fun (Callable[[InferenceResultBatch], torch.Tensor]): The loss function to calculate the loss
            num_train_steps_done (int): The number of training steps done so far for logging purposes

        Returns:
            dict[str, EvaluationResultBatch]: A dictionary containing the evaluation results for each dataloader
        """
        result_dict: dict[str, EvaluationResultBatch] = {}
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for data_loader in data_loaders:
            cumulated_loss = torch.zeros(3).to(device)

            Evaluator._publish_progress(
                progress_publisher=self.progress_publisher,
                num_eval_steps_done=0,  # Reset progress bar
                dataloader_tag=data_loader.dataloader_tag,
            )
            thoughput_aggregator = Aggregator[ThroughputAggregationKeys]()
            with TimeRecorder() as forward_backward_timer_recorder:
                for batch_id, batch in enumerate(data_loader):
                    batch_loss = self.evaluate_batch(
                        batch=batch,
                        model=model,
                        loss_fun=loss_fun,
                    )

                    cumulated_loss[0] += batch_loss.item()  # sum up batch loss
                    cumulated_loss[1] += 1
                    batch_length_tensor = torch.tensor(len(batch)).to(device)
                    thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)

                    Evaluator._publish_progress(
                        progress_publisher=self.progress_publisher,
                        num_eval_steps_done=batch_id + 1,
                        dataloader_tag=data_loader.dataloader_tag,
                    )
            # TODO: insert reducer from outside so Evaluator is independent of FSDP
            total_loss = Reducer.reduce(
                tensor=cumulated_loss,
                operation=dist.ReduceOp.SUM,
                post_processing_fun=lambda t: t[0] / t[1],
            )

            forward_backward_time = torch.tensor(forward_backward_timer_recorder.delta_t).to(device)
            thoughput_aggregator.add_value(
                key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_time
            )
            synced_num_samples = thoughput_aggregator.get_all_reduced_value(ThroughputAggregationKeys.NUM_SAMPLES)
            synced_forward_backward_time = thoughput_aggregator.get_all_reduced_value(
                ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
            )
            num_samples_per_second = synced_num_samples / synced_forward_backward_time

            evaluation_result = EvaluationResultBatch(
                losses={loss_fun.tag: ResultItem(total_loss, decimal_places=2)},
                # TODO: hardcoded metric key
                throughput_metrics={
                    "evaluation_num_samples_per_second": ResultItem(num_samples_per_second, decimal_places=1)
                },
                dataloader_tag=data_loader.dataloader_tag,
                num_train_steps_done=num_train_steps_done,
            )
            Evaluator._publish_evaluation_result(
                evaluation_result_publisher=self.evaluation_result_publisher,
                evaluation_result=evaluation_result,
            )
            result_dict[data_loader.dataloader_tag] = evaluation_result

        model.train()

        return result_dict

    @staticmethod
    def _publish_progress(
        progress_publisher: MessagePublisher[ProgressUpdate],
        num_eval_steps_done: int,
        dataloader_tag: str,
    ):
        payload = ProgressUpdate(
            num_steps_done=num_eval_steps_done,
            experiment_status=ExperimentStatus.EVALUATION,
            dataloader_tag=dataloader_tag,
        )
        progress_publisher.publish_message(payload=payload, message_type=MessageTypes.BATCH_PROGRESS_UPDATE)

    @staticmethod
    def _publish_evaluation_result(
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        evaluation_result: EvaluationResultBatch,
    ):
        evaluation_result_publisher.publish_message(
            payload=evaluation_result, message_type=MessageTypes.EVALUATION_RESULT
        )
