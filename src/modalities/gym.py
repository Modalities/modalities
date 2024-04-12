from functools import partial
from typing import List

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.checkpointing.checkpoint_saving import CheckpointSaving
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.evaluator import Evaluator
from modalities.loss_functions import Loss
from modalities.models.model import NNModel
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
        callback_interval_in_batches: int,
        train_data_loader: LLMDataLoader,
        evaluation_data_loaders: List[LLMDataLoader],
        checkpoint_saving: CheckpointSaving,
    ):
        self._run_evaluation_and_checkpointing(
            model=model,
            optimizer=optimizer,
            # here, fast_forward_sample_id points to the next sample_id that we would
            # perform forward over. Therefore, -1 one for the current sample_id.
            local_train_sample_id=train_data_loader.fast_forward_sample_id - 1,
            evaluation_data_loaders=evaluation_data_loaders,
            checkpoint_saving=checkpoint_saving,
        )

        self.trainer.train(
            model=model,
            train_loader=train_data_loader,
            loss_fun=self.loss_fun,
            optimizer=optimizer,
            scheduler=scheduler,
            callback_interval_in_batches=callback_interval_in_batches,
            epoch_done_callback=partial(  # TODO rename to something more meaningful
                self._run_evaluation_and_checkpointing,
                model=model,
                optimizer=optimizer,
                evaluation_data_loaders=evaluation_data_loaders,
                checkpoint_saving=checkpoint_saving,
            ),
            local_sample_id_to_global_sample_id=self._local_sample_id_to_global_sample_id,
        )

    def _run_evaluation_and_checkpointing(
        self,
        model: NNModel,
        optimizer: Optimizer,
        local_train_sample_id: int,
        evaluation_data_loaders: List[LLMDataLoader],
        checkpoint_saving: CheckpointSaving,
    ):
        global_train_sample_id = self._local_sample_id_to_global_sample_id(local_sample_id=local_train_sample_id)

        eval_result = self.evaluator.evaluate(
            model=model,
            data_loaders=evaluation_data_loaders,
            loss_fun=self.loss_fun,
            global_train_sample_id=global_train_sample_id,
            local_sample_id_to_global_sample_id=self._local_sample_id_to_global_sample_id,
        )

        # TODO: implement early stopping
        checkpoint_saving.save_checkpoint(
            global_train_sample_id=global_train_sample_id,
            evaluation_result=eval_result,
            model=model,
            optimizer=optimizer,
            early_stoppping_criterion_fulfilled=False,
        )

    def _local_sample_id_to_global_sample_id(self, local_sample_id: int) -> int:
        """Calculates the global sample id as an aggregation over all ranks

        Args:
            local_sample_id (int): sample id for a given rank

        Returns:
            int: global sample id
        """
        return (local_sample_id + 1) * self.num_ranks - 1

    def _local_num_samples_to_global_num_samples(self, local_num_samples: int) -> int:
        """Calculates the number of samples across all ranks.

        Args:
            local_num_samples (int): num samples per rank
            num_ranks (int): number of ranks

        Returns:
            int: number of samples summed over all ranks
        """
        return local_num_samples * self.num_ranks
