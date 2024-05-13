from functools import partial
from typing import Callable, List

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.checkpointing.checkpoint_saving import CheckpointSaving
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.loops.evaluation.evaluation_loop import EvaluationLoop
from modalities.loops.training.training_loop import TrainingLoop
from modalities.loss_functions import Loss


class Gym:
    def __init__(
        self, training_loop: TrainingLoop, evaluation_loop: EvaluationLoop, loss_fun: Loss, num_ranks: int
    ) -> None:
        self.training_loop = training_loop
        self.evaluation_loop = evaluation_loop
        self.loss_fun = loss_fun
        self.num_ranks = num_ranks

    def run(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        global_checkpointing_interval_in_steps: int,
        global_evaluation_interval_in_steps: int,
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
            global_evaluation_interval_in_steps=global_evaluation_interval_in_steps,
        )

        checkpointing_callback: Callable[[int], None] = partial(
            self._run_checkpointing,
            model=model,
            optimizer=optimizer,
            checkpoint_saving=checkpoint_saving,
            global_checkpointing_interval_in_steps=global_checkpointing_interval_in_steps,
        )

        self.training_loop.train(
            model=model,
            train_loader=train_data_loader,
            loss_fun=self.loss_fun,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluation_callback=evaluation_callback,
            checkpointing_callback=checkpointing_callback,
        )

    def _run_checkpointing(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_step_id: int,
        checkpoint_saving: CheckpointSaving,
        global_checkpointing_interval_in_steps: int,
    ):
        if (train_step_id + 1) % global_checkpointing_interval_in_steps == 0:
            checkpoint_saving.save_checkpoint(
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
        if (train_step_id + 1) % global_evaluation_interval_in_steps == 0:
            self.evaluation_loop.evaluate(
                model=model,
                data_loaders=evaluation_data_loaders,
                loss_fun=self.loss_fun,
                train_step_id=train_step_id,
            )
