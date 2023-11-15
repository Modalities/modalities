from functools import partial
from typing import List
from llm_gym.checkpointing.checkpointing import Checkpointing
from llm_gym.dataset_loader import LLMDataLoader, RepeatingDataLoader
from llm_gym.evaluator import Evaluator
from llm_gym.models.model import NNModel
from llm_gym.loss_functions import Loss
from llm_gym.trainer import Trainer
from torch.optim import Optimizer


class Gym:
    def __init__(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        loss_fun: Loss,
    ) -> None:
        self.trainer = trainer
        self.evaluator = evaluator
        self.loss_fun = loss_fun

    def run(
        self,
        model: NNModel,
        optimizer: Optimizer,
        num_batches: int,
        num_batches_per_epoch: int,
        train_data_loader: LLMDataLoader,
        evaluation_data_loaders: List[LLMDataLoader],
        checkpointing: Checkpointing,
    ):
        self._run_evaluation_and_checkpointing(
            model=model,
            optimizer=optimizer,
            num_batches=num_batches,
            train_batch_id=-1,
            evaluation_data_loaders=evaluation_data_loaders,
            checkpointing=checkpointing,
        )

        train_repeating_data_loader = RepeatingDataLoader(data_loader=train_data_loader)
        self.trainer.train(
            model=model,
            train_loader=train_repeating_data_loader,
            loss_fun=self.loss_fun,
            optimizer=optimizer,
            num_batches_per_epoch=num_batches_per_epoch,
            num_batches=num_batches,
            epoch_done_callback=partial(
                self._run_evaluation_and_checkpointing,
                model=model,
                optimizer=optimizer,
                evaluation_data_loaders=evaluation_data_loaders,
                num_batches=num_batches,
                checkpointing=checkpointing,
            ),
        )

    def _run_evaluation_and_checkpointing(
        self,
        model: NNModel,
        optimizer: Optimizer,
        num_batches: int,
        train_batch_id: int,
        evaluation_data_loaders: List[LLMDataLoader],
        checkpointing: Checkpointing,
    ):
        eval_result = self.evaluator.evaluate_epoch(
            model=model,
            data_loaders=evaluation_data_loaders,
            loss_fun=self.loss_fun,
            train_batch_id=train_batch_id,
        )

        # TODO: implement early stopping
        checkpointing.save_checkpoint(
            train_batch_id=train_batch_id,
            num_batches=num_batches,
            evaluation_result=eval_result,
            model=model,
            optimizer=optimizer,
            early_stoppping_criterion_fulfilled=False,
        )
