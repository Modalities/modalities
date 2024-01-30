from typing import Callable, Dict, List

import torch
import torch.distributed as dist

from modalities.batch import DatasetBatch, EvaluationResultBatch, InferenceResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.models.model import NNModel, model_predict_batch
from modalities.running_env.fsdp.reducer import Reducer
from modalities.trainer import ThroughputAggregationKeys
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
        global_train_sample_id: int,
        local_sample_id_to_global_sample_id: Callable[[int], int],
    ) -> Dict[str, EvaluationResultBatch]:
        result_dict: Dict[str, EvaluationResultBatch] = {}
        model.eval()

        device = torch.device(self.local_rank) if torch.cuda.is_available() else "cpu"

        for data_loader in data_loaders:
            cummulated_loss = torch.zeros(3).to(device)

            Evaluator._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                global_train_sample_id=global_train_sample_id,
                global_dataset_sample_id=-1,
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

                    cummulated_loss[0] += batch_loss.item()  # sum up batch loss
                    cummulated_loss[1] += len(batch)
                    batch_length_tensor = torch.tensor(len(batch)).to(device)
                    thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)

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
            # TODO: insert reducer from outside so Evaluator is independent of FSDP
            total_loss = Reducer.reduce(
                tensor=cummulated_loss,
                operation=dist.ReduceOp.SUM,
                post_processing_fun=lambda t: t[0] / t[1],
            )

            foward_backward_time = torch.tensor(forward_backward_timer_recorder.delta_t).to(device)
            thoughput_aggregator.add_value(
                key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=foward_backward_time
            )
            synced_num_samples = thoughput_aggregator.get_all_reduced_value(ThroughputAggregationKeys.NUM_SAMPLES)
            synced_foward_backward_time = thoughput_aggregator.get_all_reduced_value(
                ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
            )
            num_samples_per_second = synced_num_samples / synced_foward_backward_time

            evaluation_result = EvaluationResultBatch(
                losses={loss_fun.tag: total_loss},
                # TODO: hardcoded metric key
                throughput_metrics={"evaluation_num_samples_per_second": num_samples_per_second},
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
