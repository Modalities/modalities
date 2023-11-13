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
        checkpointing: Checkpointing,
        trainer: Trainer,
        evaluator: Evaluator,
        model: NNModel,
        optimizer: Optimizer,
        loss_fun: Loss,
    ) -> None:
        self.checkpointing = checkpointing
        self.trainer = trainer
        self.evaluator = evaluator
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun

    def run(
        self,
        num_batches: int,
        num_batches_per_epoch: int,
        train_data_loader: LLMDataLoader,
        evaluation_data_loaders: List[LLMDataLoader],
    ):
        self._run_evaluation_and_checkpointing(
            num_batches=num_batches, train_batch_id=-1, evaluation_data_loaders=evaluation_data_loaders
        )

        train_repeating_data_loader = RepeatingDataLoader(data_loader=train_data_loader)
        self.trainer.train(
            model=self.model,
            train_loader=train_repeating_data_loader,
            loss_fun=self.loss_fun,
            optimizer=self.optimizer,
            num_batches_per_epoch=num_batches_per_epoch,
            num_batches=num_batches,
            epoch_done_callback=partial(
                self._run_evaluation_and_checkpointing,
                evaluation_data_loaders=evaluation_data_loaders,
                num_batches=num_batches,
            ),
        )

    def _run_evaluation_and_checkpointing(
        self,
        num_batches: int,
        train_batch_id: int,
        evaluation_data_loaders: List[LLMDataLoader],
    ):
        eval_result = self.evaluator.evaluate_epoch(
            model=self.model,
            data_loaders=evaluation_data_loaders,
            loss_fun=self.loss_fun,
            train_batch_id=train_batch_id,
        )

        # TODO: implement early stopping
        self.checkpointing.run(
            train_batch_id=train_batch_id,
            num_batches=num_batches,
            evaluation_result=eval_result,
            model=self.model,
            early_stoppping_criterion_fulfilled=False,
        )
