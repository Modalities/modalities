from datetime import datetime
from functools import partial
from typing import Callable, List

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.checkpointing.checkpoint_saving import CheckpointSaving
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.evaluator import Evaluator
from modalities.loss_functions import Loss
from modalities.trainer import Trainer
from modalities.util import print_rank_0


class Gym:
    def __init__(self, trainer: Trainer, evaluator: Evaluator, loss_fun: Loss, num_ranks: int) -> None:
        self.trainer = trainer
        self.evaluator = evaluator
        self.loss_fun = loss_fun
        self.num_ranks = num_ranks

    def run(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        training_log_interval_in_steps: int,
        checkpointing_interval_in_steps: int,
        evaluation_interval_in_steps: int,
        train_data_loader: LLMDataLoader,
        evaluation_data_loaders: List[LLMDataLoader],
        checkpoint_saving: CheckpointSaving,
    ):
        # self._run_evaluation(
        #     model=model,
        #     # here, fast_forward_sample_id points to the next sample_id that we would
        #     # perform forward over. Therefore, -1 one for the current sample_id.
        #     local_train_sample_id=train_data_loader.fast_forward_sample_id - 1,
        #     local_evaluation_interval_in_samples=local_evaluation_interval_in_samples,
        #     evaluation_data_loaders=evaluation_data_loaders,
        #     checkpoint_saving=checkpoint_saving,
        # )
        evaluation_callback: Callable[[int], None] = partial(
            self._run_evaluation,
            model=model,
            evaluation_data_loaders=evaluation_data_loaders,
            evaluation_interval_in_steps=evaluation_interval_in_steps,
        )

        checkpointing_callback: Callable[[int], None] = partial(
            self._run_checkpointing,
            model=model,
            optimizer=optimizer,
            checkpoint_saving=checkpoint_saving,
            checkpointing_interval_in_steps=checkpointing_interval_in_steps,
        )

        print_rank_0(f"Start model training at {datetime.now()}.")
        self.trainer.train(
            model=model,
            train_loader=train_data_loader,
            loss_fun=self.loss_fun,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluation_callback=evaluation_callback,
            checkpointing_callback=checkpointing_callback,
            training_log_interval_in_steps=training_log_interval_in_steps,
        )

    def _run_checkpointing(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        num_train_steps_done: int,
        checkpoint_saving: CheckpointSaving,
        checkpointing_interval_in_steps: int,
    ):
        if num_train_steps_done % checkpointing_interval_in_steps == 0 and num_train_steps_done > 0:
            checkpoint_saving.save_checkpoint(
                num_train_steps_done=num_train_steps_done,
                evaluation_result=None,  # TODO implement checkpointing based on preceding evaluation results
                model=model,
                optimizer=optimizer,
                early_stoppping_criterion_fulfilled=False,  # TODO: implement early stopping
            )

    def _run_evaluation(
        self,
        model: nn.Module,
        num_train_steps_done: int,
        evaluation_data_loaders: List[LLMDataLoader],
        evaluation_interval_in_steps: int,
    ):
        if num_train_steps_done % evaluation_interval_in_steps == 0:
            self.evaluator.evaluate(
                model=model,
                data_loaders=evaluation_data_loaders,
                loss_fun=self.loss_fun,
                num_train_steps_done=num_train_steps_done,
            )
