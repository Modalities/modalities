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
from modalities.training.training_progress import TrainingProgress
from modalities.util import print_rank_0


class Gym:
    """Class to perform the model training, including evaluation and checkpointing."""

    def __init__(self, trainer: Trainer, evaluator: Evaluator, loss_fun: Loss, num_ranks: int) -> None:
        """Initializes a Gym object.

        Args:
            trainer (Trainer): Trainer object to perform the training.
            evaluator (Evaluator): Evaluator object to perform the evaluation.
            loss_fun (Loss): Loss function applied during training and evaluation.
            num_ranks (int): Number of ranks used for distributed training.
        """
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
        """Runs the model training, including evaluation and checkpointing.

        Args:
            model (nn.Module): Model to be trained.
            optimizer (Optimizer): Optimizer used for training.
            scheduler (LRScheduler): Scheduler used for training.
            training_log_interval_in_steps (int): Interval in steps to log training progress.
            checkpointing_interval_in_steps (int): Interval in steps to save checkpoints.
            evaluation_interval_in_steps (int): Interval in steps to perform evaluation.
            train_data_loader (LLMDataLoader): Data loader with the training data.
            evaluation_data_loaders (List[LLMDataLoader]): List of data loaders with the evaluation data.
            checkpoint_saving (CheckpointSaving): Routine for saving checkpoints.
        """
        evaluation_callback: Callable[[int], None] = partial(
            self._run_evaluation,
            model=model,
            evaluation_data_loaders=evaluation_data_loaders,
            evaluation_interval_in_steps=evaluation_interval_in_steps,
        )

        checkpointing_callback: Callable[[TrainingProgress], None] = partial(
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
        print_rank_0(f"Training done at {datetime.now()}.")

    def _run_checkpointing(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        training_progress: TrainingProgress,
        checkpoint_saving: CheckpointSaving,
        checkpointing_interval_in_steps: int,
    ):
        if (
            training_progress.num_seen_steps_total % checkpointing_interval_in_steps == 0
            and training_progress.num_seen_steps_total > 0
        ):
            checkpoint_saving.save_checkpoint(
                training_progress=training_progress,
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
