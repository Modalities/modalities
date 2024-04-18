from functools import partial
from typing import Callable, List

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.checkpointing.checkpointing import Checkpointing
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.evaluator import Evaluator
from modalities.loss_functions import Loss
from modalities.trainer import Trainer


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
        global_training_log_interval_in_steps: int,
        global_checkpointing_interval_in_steps: int,
        global_evaluation_interval_in_steps: int,
        train_data_loader: LLMDataLoader,
        evaluation_data_loaders: List[LLMDataLoader],
        checkpointing: Checkpointing,
    ):
        # self._run_evaluation(
        #     model=model,
        #     # here, fast_forward_sample_id points to the next sample_id that we would
        #     # perform forward over. Therefore, -1 one for the current sample_id.
        #     local_train_sample_id=train_data_loader.fast_forward_sample_id - 1,
        #     local_evaluation_interval_in_samples=local_evaluation_interval_in_samples,
        #     evaluation_data_loaders=evaluation_data_loaders,
        # )
        evaluation_callback: Callable[[int], None] = partial(
            self._run_evaluation,
            model=model,
            evaluation_data_loaders=evaluation_data_loaders,
            global_evaluation_interval_in_steps=global_evaluation_interval_in_steps,
        )

        checkpointing_callback: Callable[[int], None] = partial(
            self._run_checkpointing,
            model=model,
            optimizer=optimizer,
            checkpointing=checkpointing,
            global_checkpointing_interval_in_steps=global_checkpointing_interval_in_steps,
        )

        self.trainer.train(
            model=model,
            train_loader=train_data_loader,
            loss_fun=self.loss_fun,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluation_callback=evaluation_callback,
            checkpointing_callback=checkpointing_callback,
            global_training_log_interval_in_steps=global_training_log_interval_in_steps,
        )

    def _run_checkpointing(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_step_id: int,
        checkpointing: Checkpointing,
        global_checkpointing_interval_in_steps: int,
    ):
        if (train_step_id + 1) % global_checkpointing_interval_in_steps == 0:
            checkpointing.save_checkpoint(
                train_step_id=train_step_id,
                evaluation_result=None,  # TODO implement checkpointing based on preceding evaluation results
                model=model,
                optimizer=optimizer,
                early_stoppping_criterion_fulfilled=False,  # TODO: implement early stopping
            )

    def _run_evaluation(
        self,
        model: nn.Module,
        train_step_id: int,
        evaluation_data_loaders: List[LLMDataLoader],
        global_evaluation_interval_in_steps: int,
    ):
        if (train_step_id) % global_evaluation_interval_in_steps == 0:
            self.evaluator.evaluate(
                model=model,
                data_loaders=evaluation_data_loaders,
                loss_fun=self.loss_fun,
                train_step_id=train_step_id,
            )
