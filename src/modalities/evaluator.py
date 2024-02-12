from typing import Callable, Dict, List, Tuple

import torch

from modalities.batch import DatasetBatch, EvaluationResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.evaluation.measure import AggregativeMeasure, AggregativeMeasureFactory
from modalities.evaluation.throughput_aggregator import ThroughputAggregationContext, ThroughputAggregator
from modalities.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.models.model import NNModel, model_predict_batch


class Evaluator:
    def __init__(
        self,
        local_rank: int,
        loss_factories: List[AggregativeMeasureFactory],
        metric_factories: List[AggregativeMeasureFactory],
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        throughput_aggregator_factory: Callable[[], ThroughputAggregator],
    ) -> None:
        self.local_rank = local_rank
        self._loss_factories = loss_factories
        self._metric_factories = metric_factories
        self.batch_progress_publisher = batch_progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self._throughput_aggregator_factory = throughput_aggregator_factory

    def evaluate(
        self,
        model: NNModel,
        data_loaders: List[LLMDataLoader],
        global_train_sample_id: int,
        local_sample_id_to_global_sample_id: Callable[[int], int],
    ) -> Dict[str, EvaluationResultBatch]:
        result_dict: Dict[str, EvaluationResultBatch] = {}
        model.eval()
        for data_loader in data_loaders:
            Evaluator._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                global_train_sample_id=global_train_sample_id,
                global_dataset_sample_id=-1,
                dataloader_tag=data_loader.dataloader_tag,
            )  # TODO why is this in the beginning of the for loop, not at the end?

            with ThroughputAggregationContext(
                len(data_loader), self.local_rank, self._throughput_aggregator_factory
            ) as thoughput_agg:
                total_losses, total_metrics = self._process_batches_in_data_loader(
                    model=model,
                    data_loader=data_loader,
                    global_train_sample_id=global_train_sample_id,
                    local_sample_id_to_global_sample_id=local_sample_id_to_global_sample_id,
                )

            evaluation_result = EvaluationResultBatch(
                losses=total_losses,
                metrics=total_metrics,
                # TODO: hardcoded metric key
                throughput_metrics={"evaluation_num_samples_per_second": thoughput_agg.samples_per_second},
                dataloader_tag=data_loader.dataloader_tag,
                global_train_sample_id=global_train_sample_id,
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

    def _process_batches_in_data_loader(
        self,
        model: NNModel,
        data_loader: LLMDataLoader,
        global_train_sample_id: int,
        local_sample_id_to_global_sample_id: Callable[[int], int],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        losses = [f.create(self.local_rank) for f in self._loss_factories]
        metrics = [f.create(self.local_rank) for f in self._metric_factories]

        for batch_id, batch in enumerate(data_loader):
            self.evaluate_batch(
                batch=batch,
                model=model,
                loss_functions=losses,
                metrics=metrics,
            )

            local_dataset_sample_id = Evaluator._get_local_sample_id(
                batch_id=batch_id, batch_size=data_loader.sampler_batch_size
            )
            global_dataset_sample_id = local_sample_id_to_global_sample_id(local_dataset_sample_id)
            Evaluator._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                global_train_sample_id=global_train_sample_id,
                global_dataset_sample_id=global_dataset_sample_id,
                dataloader_tag=data_loader.dataloader_tag,
            )

        return {loss: loss.compute() for loss in losses}, {metric: metric.compute() for metric in metrics}

    def evaluate_batch(
        self,
        batch: DatasetBatch,
        model: NNModel,
        loss_functions: List[AggregativeMeasure],
        metrics: List[AggregativeMeasure],
    ) -> None:
        with torch.no_grad():
            result_batch = model_predict_batch(model=model, batch=batch)
        for loss_fun in loss_functions:
            loss_fun.add(result_batch)
        for metric_fun in metrics:
            metric_fun.add(result_batch)

    @staticmethod
    def _publish_progress(
        batch_progress_publisher: MessagePublisher[BatchProgressUpdate],
        global_train_sample_id: int,
        global_dataset_sample_id: int,
        dataloader_tag: str,
    ):
        payload = BatchProgressUpdate(
            global_train_sample_id=global_train_sample_id,
            global_dataset_sample_id=global_dataset_sample_id,
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

    @staticmethod
    def _get_local_sample_id(batch_id: int, batch_size: int) -> int:
        return (batch_id + 1) * batch_size - 1
