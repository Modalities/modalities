from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn

from modalities.batch import DatasetBatch, EvaluationResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import model_predict_batch
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
        model: nn.Module,
        loss_fun: List[Loss],
    ):
        result_batch = model_predict_batch(model=model, batch=batch)

        total_loss = None
        losses = []
        for lfn in loss_fun:
            # Calculate loss
            weighted_loss = lfn(result_batch) * lfn.weight

            # Add loss to total loss
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss += weighted_loss

            # Append individual losses (for logging)
            losses.append(weighted_loss.clone().detach())

        return total_loss, *losses

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        data_loaders: List[LLMDataLoader],
        loss_fun: List[Loss],
        train_step_id: int,
    ) -> Dict[str, EvaluationResultBatch]:
        result_dict: Dict[str, EvaluationResultBatch] = {}
        model.eval()

        device = torch.device(self.local_rank if torch.cuda.is_available() else "cpu")

        for data_loader in data_loaders:
            cumulated_loss = torch.zeros(len(loss_fun) + 1 + 1).to(device)  # total loss, indidual losses, count

            Evaluator._publish_progress(
                batch_progress_publisher=self.batch_progress_publisher,
                eval_step_id=0,  # Reset progress bar
                dataloader_tag=data_loader.dataloader_tag,
            )
            thoughput_aggregator = Aggregator[ThroughputAggregationKeys]()

            # Make sure that all ranks reach this point at the same time
            dist.barrier()

            forward_backward_time_recorder = TimeRecorder()
            forward_backward_time_recorder.start()
            for batch_id, batch in enumerate(data_loader):
                batch_losses = self.evaluate_batch(
                    batch=batch,
                    model=model,
                    loss_fun=loss_fun,
                )
                forward_backward_time_recorder.stop()

                # Accumulate losses
                for i, batch_loss in enumerate(batch_losses):
                    cumulated_loss[i] += batch_loss.item()
                cumulated_loss[-1] += 1

                batch_length_tensor = torch.tensor(len(batch)).to(device)
                thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)

                Evaluator._publish_progress(
                    batch_progress_publisher=self.batch_progress_publisher,
                    eval_step_id=batch_id,
                    dataloader_tag=data_loader.dataloader_tag,
                )

                # we start the time recoder here again to also capture the time spend loading
                # via the dataloader.
                forward_backward_time_recorder.start()

            # TODO: insert reducer from outside so Evaluator is independent of FSDP
            forward_backward_time = torch.tensor(forward_backward_time_recorder.delta_t).to(device)
            thoughput_aggregator.add_value(
                key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_time
            )
            synced_num_samples = thoughput_aggregator.get_all_reduced_value(ThroughputAggregationKeys.NUM_SAMPLES)
            synced_forward_backward_time = thoughput_aggregator.get_all_reduced_value(
                ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
            )
            num_samples_per_second = synced_num_samples / synced_forward_backward_time

            # Agreggate loss from all ranks
            reduced_losses = Reducer.reduce(
                tensor=cumulated_loss,
                operation=dist.ReduceOp.SUM,
                post_processing_fun=lambda t: torch.cat([t[:-1] / t[-1], t[-1:] / dist.get_world_size()]),
            )

            # Fill logging dict with total loss and the individual losses
            loss_avg, loss_last_batch = (
                reduced_losses[0],
                reduced_losses[-1],
            )

            losses = {
                "total_loss average": loss_avg,
                "total_loss last step": loss_last_batch,
            }
            for i, lfn in enumerate(loss_fun):
                losses[lfn.tag] = reduced_losses[i + 1]

            evaluation_result = EvaluationResultBatch(
                losses=losses,
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

        model.train()

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
