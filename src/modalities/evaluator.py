from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist

from modalities.batch import DatasetBatch, EvaluationResultBatch
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import BatchProgressUpdate, ExperimentStatus, MessageTypes
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.metrics import Metric
from modalities.models.model import NNModel, model_predict_batch
from modalities.running_env.fsdp.reducer import Reducer


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
        loss_functions: List[Loss],
        metrics: List[Metric],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        with torch.no_grad():
            result_batch = model_predict_batch(model=model, batch=batch)
        losses = {l.tag: l(result_batch) for l in loss_functions}
        metrics = {m.tag: m(result_batch) for m in metrics}
        return losses, metrics

    def evaluate(
        self,
        model: NNModel,
        data_loaders: List[LLMDataLoader],
        loss_functions: List[Loss],
        metrics: List[Metric],
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
            )

            cummulated_losses, cummulated_metrics = self._process_batches_in_data_loader(
                model=model,
                data_loader=data_loader,
                loss_functions=loss_functions,
                metrics=metrics,
                global_train_sample_id=global_train_sample_id,
                local_sample_id_to_global_sample_id=local_sample_id_to_global_sample_id,
            )
            total_losses = self._reduce_cummulatives_vars(cummulated_losses)
            total_metrics = self._reduce_cummulatives_vars(cummulated_metrics)

            evaluation_result = EvaluationResultBatch(
                losses=total_losses,
                metrics=total_metrics,
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
        loss_functions: List[Loss],
        metrics: List[Metric],
        global_train_sample_id: int,
        local_sample_id_to_global_sample_id: Callable[[int], int],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        cummulated_losses = {l.tag: self._prepare_cummulated_var() for l in loss_functions}
        cummulated_metrics = {m.tag: self._prepare_cummulated_var() for m in metrics}

        for batch_id, batch in enumerate(data_loader):
            batch_losses, batch_metrics = self.evaluate_batch(
                batch=batch,
                model=model,
                loss_functions=loss_functions,
                metrics=metrics,
            )

            self._update_cummulated_vars(len(batch), batch_losses, cummulated_losses)
            self._update_cummulated_vars(len(batch), batch_metrics, cummulated_metrics)

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

        return cummulated_losses, cummulated_metrics

    def _prepare_cummulated_var(self) -> torch.Tensor:
        if torch.cuda.is_available():
            cummulated_loss = torch.zeros(3).to(torch.device(self.local_rank))
        else:
            cummulated_loss = torch.zeros(3).to("cpu")
        return cummulated_loss

    def _update_cummulated_vars(
        self, batch_len: int, update: Dict[str, torch.Tensor], cummulated: Dict[str, torch.Tensor]
    ):
        for tag, val in update.items():
            cummulated[tag][0] += val.item()
            cummulated[tag][1] += batch_len

    def _reduce_cummulatives_vars(self, cummulated: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            # TODO: insert reducer from outside so Evaluator is independent of FSDP
            tag: Reducer.reduce(
                tensor=val,
                operation=dist.ReduceOp.SUM,
                post_processing_fun=lambda t: t[0] / t[1],
            )
            for tag, val in cummulated.items()
        }

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
