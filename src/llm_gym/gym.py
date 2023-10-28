from abc import abstractmethod
from typing import Any, Callable, Dict
from llm_gym.forward_pass import ModelForwardPass
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from llm_gym.batch import DatasetBatch, InferenceResultBatch
from torch.optim import Optimizer
from rich.progress import Progress


class ProgressCallbackIF:

    @abstractmethod
    def __call__(self, epoch_increment: int = 0, batch_increment: int = 1, split_key: str = None) -> Any:
        raise NotImplementedError


class RichProgressCallback:
    def __init__(self, subscribing_global_rank: int, num_epochs: int, split_lengths: Dict[str, int]) -> None:
        self.subscribing_global_rank = subscribing_global_rank
        if dist.get_rank() == self.subscribing_global_rank:
            self.current_epoch = 0
            self.progress = Progress()
            self.split_keys = list(split_lengths.keys())
            self.tasks = {}
            for split_key, split_length in split_lengths.items():
                task = self.progress.add_task(description=split_key, total=split_length)
                self.tasks[split_key] = task
            task = self.progress.add_task(description="epochs", total=num_epochs)
            self.tasks["epochs"] = task   # TODO fix hardcoding

    def __call__(self, epoch_increment: int = 0, batch_increment: int = 1, split_key: str = None) -> Any:
        if dist.get_rank() == self.subscribing_global_rank:
            if epoch_increment > 0:
                self.progress.update("epochs", advance=epoch_increment)
                for sk in self.split_keys:
                    self.progress.update(self.tasks[sk], completed=0)
            else:
                self.progress.update(self.tasks[split_key], advance=batch_increment)


class ResultsCallbackIF:
    def __call__(self, evaluation_result: Dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError


class ResultsCallback:
    def __init__(self, subscribing_global_rank: int) -> None:
        self.subscribing_global_rank = subscribing_global_rank
        if dist.get_rank() == self.subscribing_global_rank:
            pass

    def __call__(self, evaluation_result: Dict[str, torch.Tensor]) -> Any:
        if dist.get_rank() == self.subscribing_global_rank:
            print(evaluation_result)


class Trainer:

    def __init__(self, local_rank: int, global_rank: int, batch_processed_callback: ProgressCallbackIF, results_callback: ResultsCallbackIF) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.batch_processed_callback = batch_processed_callback
        self.results_callback = results_callback

    def _train_batch(self, batch: DatasetBatch, model_forward_pass: ModelForwardPass,
                     optimizer: Optimizer, loss_fun: Callable[[InferenceResultBatch], torch.Tensor]) -> torch.Tensor:
        batch.to(self.local_rank)
        optimizer.zero_grad()
        result_batch = model_forward_pass.predict(batch, no_grad=False)
        loss = loss_fun(result_batch)
        loss.backward()
        optimizer.step()
        return loss

    def train_epoch(self, model_forward_pass: ModelForwardPass, train_loader: DataLoader, optimizer,
                    loss_fun: Callable[[InferenceResultBatch], torch.Tensor]):
        fsdp_loss = torch.zeros(2).to(self.local_rank)

        batch: DatasetBatch
        for batch in train_loader:
            batch_loss = self._train_batch(batch=batch, model_forward_pass=model_forward_pass, optimizer=optimizer, loss_fun=loss_fun)

            fsdp_loss[0] += batch_loss.item()
            fsdp_loss[1] += len(batch)
            self.batch_processed_callback(batch_increment=1, split_key="train")  # TODO add the split key to the dataloader class!

        train_loss = Reducer.reduce(tensor=fsdp_loss, operation=dist.ReduceOp.SUM,
                                    post_processing_fun=lambda t: t[0] / t[1])

        self.batch_processed_callback(epoch_increment=1)
        self.results_callback(evaluation_result={"train_loss": train_loss})
        return train_loss


class Evaluator:
    def __init__(self, local_rank: int, global_rank: int, batch_processed_callback: ProgressCallbackIF, results_callback: ResultsCallbackIF) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.batch_processed_callback = batch_processed_callback
        self.results_callback = results_callback

    def evaluate_batch(self, batch: DatasetBatch, model_forward_pass: ModelForwardPass, loss_fun: Callable[[InferenceResultBatch], torch.Tensor]):
        batch.to(self.local_rank)
        result_batch = model_forward_pass.predict(batch=batch, no_grad=True)
        loss = loss_fun(result_batch)
        return loss

    def evaluate_epoch(self, model_forward_pass: ModelForwardPass, data_loaders: Dict[str, DataLoader],
                       loss_fun: Callable[[InferenceResultBatch], torch.Tensor]):
        evaluation_result = {}
        for dataloader_key, data_loader in data_loaders.items():
            fsdp_loss = torch.zeros(3).to(self.local_rank)
            for batch in data_loader:
                batch_loss = self.evaluate_batch(batch=batch, model_forward_pass=model_forward_pass, loss_fun=loss_fun)

                fsdp_loss[0] += batch_loss.item()  # sum up batch loss
                fsdp_loss[1] += len(batch)
                # TODO add the split key to the dataloader class!
                self.batch_processed_callback(batch_increment=1, split_key=dataloader_key)

            dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
            total_loss = fsdp_loss[0] / fsdp_loss[1]
            evaluation_result[dataloader_key] = total_loss
        self.batch_processed_callback(epoch_increment=1)
        self.results_callback(evaluation_result=evaluation_result)


class Gym:
    pass


class Reducer:

    @staticmethod
    def reduce(tensor: torch.Tensor, operation: dist.ReduceOp, callback_fun: Callable[[torch.Tensor], None] = None,
               post_processing_fun: Callable[[torch.Tensor], torch.Tensor] = None):
        reduced_tensor = dist.all_reduce(tensor, op=operation)
        if post_processing_fun is not None:
            reduced_tensor = post_processing_fun(reduced_tensor)
        if callback_fun is not None:
            callback_fun(reduced_tensor)
        return reduced_tensor


class FSDP:
    def __init__(self) -> None:
        dist.init_process_group("nccl")

    def run():

        dist.destroy_process_group()


class Checkpointer:
    pass
