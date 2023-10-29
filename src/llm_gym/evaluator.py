from typing import Callable, Dict, List
from llm_gym.fsdp.reducer import Reducer
from llm_gym.callbacks.batch_progress_callbacks import ProgressCallbackIF
from llm_gym.callbacks.results_callbacks import ResultsCallbackIF
from llm_gym.forward_pass import ModelInferenceComponent
import torch.distributed as dist
import torch
from llm_gym.batch import DatasetBatch, InferenceResultBatch, EvaluationResultBatch
from llm_gym.dataset_loader import LLMDataLoader


class Evaluator:
    def __init__(self, local_rank: int, batch_processed_callback: ProgressCallbackIF,
                 results_callback: ResultsCallbackIF) -> None:
        self.local_rank = local_rank
        self.batch_processed_callback = batch_processed_callback
        self.results_callback = results_callback

    def evaluate_batch(self, batch: DatasetBatch, model_inference_component: ModelInferenceComponent,
                       loss_fun: Callable[[InferenceResultBatch], torch.Tensor]):
        batch.to(self.local_rank)
        result_batch = model_inference_component.predict(batch=batch, no_grad=True)
        loss = loss_fun(result_batch)
        return loss

    def evaluate_epoch(self, model_inference_component: ModelInferenceComponent, data_loaders: List[LLMDataLoader],
                       loss_fun: Callable[[InferenceResultBatch], torch.Tensor]) -> Dict[str, EvaluationResultBatch]:
        result_dict: Dict[str, EvaluationResultBatch] = {}
        for data_loader in data_loaders:
            cummulated_loss = torch.zeros(3).to(self.local_rank)
            for batch in data_loader:
                batch_loss = self.evaluate_batch(batch=batch, model_inference_component=model_inference_component, loss_fun=loss_fun)

                cummulated_loss[0] += batch_loss.item()  # sum up batch loss
                cummulated_loss[1] += len(batch)
                # TODO add the split key to the dataloader class!
                self.batch_processed_callback(batch_increment=1, split_key=data_loader.dataset_tag)
            # TODO: insert reducer from outside so Evaluator is independent of FSDP
            total_loss = Reducer.reduce(tensor=cummulated_loss, operation=dist.ReduceOp.SUM,
                                        post_processing_fun=lambda t: t[0] / t[1])
            evaluation_result = EvaluationResultBatch(losses={loss_fun.tag: total_loss}, split_name=data_loader.dataset_tag)
            self.results_callback(evaluation_result=result_dict)
            result_dict[data_loader.dataset_tag] = evaluation_result
        self.batch_processed_callback(epoch_increment=1)
        return result_dict