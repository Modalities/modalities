from typing import Callable, Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn

from modalities.batch import DatasetBatch, EvaluationResultBatch, InferenceResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.models.model import model_predict_batch
from modalities.trainer import AggregationKeys
from modalities.util import Aggregator, TimeRecorder


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
        model: nn.Module,
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
    ):
        with torch.no_grad():
            result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch)
        return loss

    def evaluate(
        self,
        model: nn.Module,
        data_loaders: List[LLMDataLoader],
        loss_fun: Callable[[InferenceResultBatch], torch.Tensor],
        train_step_id: int,
    ) -> Dict[str, EvaluationResultBatch]:
        result_dict: Dict[str, EvaluationResultBatch] = {}
        model.eval()

        for data_loader in data_loaders:
            Evaluator._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                eval_step_id=0,  # Reset progress bar
                dataloader_tag=data_loader.dataloader_tag,
            )
            score_aggregator = Aggregator[AggregationKeys]()
            with TimeRecorder() as forward_backward_timer_recorder:
                for batch_id, batch in enumerate(data_loader):
                    batch_loss = self.evaluate_batch(
                        batch=batch,
                        model=model,
                        loss_fun=loss_fun,
                    )

                    score_aggregator.add_value(key=AggregationKeys.CUMM_LOSS, value=batch_loss.item())
                    # This works, because we always drop the last batch in case it has less samples than the batch size
                    score_aggregator.add_value(key=AggregationKeys.NUM_STEPS, value=1)

                    score_aggregator.add_value(key=AggregationKeys.NUM_SAMPLES, value=len(batch))

                    Evaluator._publish_progress(
                        batch_progress_publisher=self.batch_progress_publisher,
                        eval_step_id=batch_id,
                        dataloader_tag=data_loader.dataloader_tag,
                    )

            score_aggregator.add_value(
                key=AggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_timer_recorder.delta_t
            )

            # reduce the scores with the respective reduction operation
            sum_reduced_scores = score_aggregator.get_all_reduced_values(
                keys=[AggregationKeys.NUM_SAMPLES, AggregationKeys.NUM_STEPS, AggregationKeys.CUMM_LOSS],
                reduce_operation=dist.ReduceOp.SUM,
            )

            max_reduced_scores = score_aggregator.get_all_reduced_values(
                keys=[AggregationKeys.FORWARD_BACKWARD_TIME], reduce_operation=dist.ReduceOp.MAX
            )

            # calculate the metric scores for logging
            num_samples_per_second = (
                sum_reduced_scores[AggregationKeys.NUM_SAMPLES]
                / max_reduced_scores[AggregationKeys.FORWARD_BACKWARD_TIME]
            )
            eval_loss_avg = (
                sum_reduced_scores[AggregationKeys.CUMM_LOSS] / sum_reduced_scores[AggregationKeys.NUM_STEPS]
            )

            evaluation_result = EvaluationResultBatch(
                losses={loss_fun.tag: eval_loss_avg},
                # TODO: hardcoded metric key
                throughput_metrics={"evaluation_num_samples_per_second": num_samples_per_second},
                dataloader_tag=data_loader.dataloader_tag,
                train_step_id=train_step_id,
            )
            Evaluator._publish_evaluation_result(
                evaluation_result_publisher=self.evaluation_result_publisher,
                evaluation_result=evaluation_result,
            )
            result_dict[data_loader.dataloader_tag] = evaluation_result
        return result_dict

    @staticmethod
    def _publish_progress(
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        eval_step_id: int,
        dataloader_tag: str,
    ):
        payload = BatchProgressUpdate(
            step_id=eval_step_id,
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
