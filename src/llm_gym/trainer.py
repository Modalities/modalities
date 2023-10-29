from llm_gym.callbacks.batch_progress_callbacks import ProgressCallbackIF
from llm_gym.callbacks.results_callbacks import ResultsCallbackIF
from llm_gym.forward_pass import ModelInferenceComponent
from llm_gym.fsdp.reducer import Reducer
from llm_gym.loss_functions import Loss
import torch.distributed as dist
import torch
from llm_gym.batch import DatasetBatch, EvaluationResultBatch
from torch.optim import Optimizer
from llm_gym.dataset_loader import LLMDataLoader


class Trainer:

    def __init__(self, local_rank: int, batch_processed_callback: ProgressCallbackIF,
                 results_callback: ResultsCallbackIF) -> None:
        self.local_rank = local_rank
        self.batch_processed_callback = batch_processed_callback
        self.results_callback = results_callback

    def _train_batch(self, batch: DatasetBatch, model_inference_component: ModelInferenceComponent,
                     optimizer: Optimizer, loss_fun: Loss) -> torch.Tensor:
        batch.to(self.local_rank)
        optimizer.zero_grad()
        result_batch = model_inference_component.predict(batch, no_grad=False)
        loss = loss_fun(result_batch)
        loss.backward()
        optimizer.step()
        return loss

    def train_epoch(self, model_inference_component: ModelInferenceComponent, train_loader: LLMDataLoader, optimizer,
                    loss_fun: Loss):
        cummulated_loss = torch.zeros(2).to(self.local_rank)

        batch: DatasetBatch
        for batch in train_loader:
            batch_loss = self._train_batch(batch=batch, model_inference_component=model_inference_component, optimizer=optimizer, loss_fun=loss_fun)

            cummulated_loss[0] += batch_loss.item()
            cummulated_loss[1] += len(batch)
            # TODO add the split key to the dataloader class! change to train split key
            self.batch_processed_callback(batch_increment=1, split_key=train_loader.dataset_tag)

        # TODO: insert reducer from outside so Trainer is independent of FSDP
        train_loss = Reducer.reduce(tensor=cummulated_loss, operation=dist.ReduceOp.SUM,
                                    post_processing_fun=lambda t: t[0] / t[1])

        evaluation_result = EvaluationResultBatch(losses={loss_fun.tag: train_loss}, split_name=train_loader.dataset_tag)
        self.batch_processed_callback(epoch_increment=1)
        self.results_callback(evaluation_result=evaluation_result)
        return evaluation_result