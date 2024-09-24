from enum import Enum
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, EvaluationResultBatch, ResultItem
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import ExperimentStatus, MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import model_predict_batch
from modalities.running_env.fsdp.reducer import Reducer
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF
from modalities.training.training_progress import TrainingProgress
from modalities.util import Aggregator, TimeRecorder, print_rank_0
from modalities.utils.mfu import compute_mfu, get_theoretical_flops_per_token, get_theoretical_gpu_peak_performance


class ThroughputAggregationKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"


class Trainer:
    def __init__(
        self,
        global_rank: int,
        progress_publisher: MessagePublisher[ProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        gradient_acc_steps: int,
        global_num_tokens_per_train_step: int,
        num_seen_train_steps: int,
        global_num_seen_tokens: int,
        num_target_steps: int,
        num_target_tokens: int,
        gradient_clipper: GradientClipperIF,
    ) -> None:
        """
        Initializes the Trainer object.

        Args:
            global_rank (int): The global rank to which operates the trainer object.
            progress_publisher (MessagePublisher[ProgressUpdate]): The publisher for progress updates.
            evaluation_result_publisher (MessagePublisher[EvaluationResultBatch]):
                The publisher for evaluation result batches.
            gradient_acc_steps (int): The number of gradient accumulation steps.
            global_num_tokens_per_train_step (int): The number of global tokens per training step.
            target_train_steps (int): The target number of training steps.
            gradient_clipper (GradientClipperIF): The gradient clipper.

        Returns:
            None
        """
        self.global_rank = global_rank
        self.progress_publisher = progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self.gradient_acc_steps = gradient_acc_steps
        self.global_num_tokens_per_train_step = global_num_tokens_per_train_step
        self.num_seen_train_steps = num_seen_train_steps
        self.num_target_steps = num_target_steps
        self.num_target_tokens = num_target_tokens
        self.global_num_seen_tokens = global_num_seen_tokens
        self.gradient_clipper = gradient_clipper

    @staticmethod
    def _get_num_train_steps_done(micro_batch_id: int, gradient_acc_steps: int) -> int:
        """
        Calculates the number of training steps done based on the micro batch ID and gradient accumulation steps.

        Args:
            micro_batch_id (int): The ID of the current micro batch.
            gradient_acc_steps (int): The number of gradient accumulation steps.

        Returns:
            int: The number of training steps done.
        """
        return (micro_batch_id + 1) // gradient_acc_steps

    def _train_batch(
        self,
        batch: DatasetBatch,
        model: FSDP,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        micro_batch_id: int,
    ) -> Tuple[bool, int, torch.Tensor, Optional[torch.Tensor]]:
        """
        Conducts a training step on batch of data.

        Args:
            batch (DatasetBatch): The input batch of data.
            model (FSDP): The model to train.
            optimizer (Optimizer): The optimizer used for training.
            scheduler (LRScheduler): The learning rate scheduler.
            loss_fun (Loss): The loss function used for training.
            micro_batch_id (int): The ID of the micro batch.

        Returns:
            Tuple[bool, int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                A tuple containing the following:
                    - step_performed (bool): Indicates whether a training step was performed.
                    - num_train_steps_done (int): The number of training steps done.
                    - loss (torch.Tensor): The computed loss.
                    - gradient_norm_score (Optional[torch.Tensor]): The gradient norm score,
                        if a training step was performed otherwise return None.
        """
        result_batch = model_predict_batch(model=model, batch=batch)
        loss = loss_fun(result_batch)
        (loss / self.gradient_acc_steps).backward()

        if (micro_batch_id + 1) % self.gradient_acc_steps == 0:
            gradient_norm_score = self.gradient_clipper.clip_gradients()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step_performed = True
        else:
            step_performed = False
            gradient_norm_score = None

        num_train_steps_done = Trainer._get_num_train_steps_done(
            micro_batch_id=micro_batch_id, gradient_acc_steps=self.gradient_acc_steps
        )
        return step_performed, num_train_steps_done, loss, gradient_norm_score

    def train(
        self,
        model: nn.Module,
        train_loader: LLMDataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        training_log_interval_in_steps: int,
        evaluation_callback: Callable[[TrainingProgress], None],
        checkpointing_callback: Callable[[TrainingProgress], None],
    ):
        """
        Trains the model.

        Args:
            model (nn.Module): The model to be trained.
            train_loader (LLMDataLoader): The data loader containing the training data.
            optimizer (Optimizer): The optimizer used for gradient updates.
            scheduler (LRScheduler): The learning rate scheduler.
            loss_fun (Loss): The loss function used for training.
            training_log_interval_in_steps (int): The interval at which training progress is logged.
            evaluation_callback (Callable[[TrainingProgress], None]): A callback function for evaluation.
            checkpointing_callback (Callable[[TrainingProgress], None]): A callback function for checkpointing.

        Returns:
            None
        """
        model.train()
        cumulated_losses = self._reset_tracked_losses()

        # throughput & MFU
        thoughput_aggregator = Aggregator[ThroughputAggregationKeys]()
        theoretical_gpu_peak_performance = get_theoretical_gpu_peak_performance(model, world_size=dist.get_world_size())
        theoretical_flops_per_token, sequence_length = get_theoretical_flops_per_token(model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # batch loop
        batch: DatasetBatch
        # TODO: why do we need a barrier here?
        dist.barrier()
        forward_backward_time_recorder = TimeRecorder()
        forward_backward_time_recorder.start()
        gradient_norm_scores = []

        # run evaluation callback and checkpointing callback before the first optimizer step
        evaluation_callback(num_train_steps_done=self.num_seen_train_steps)
        training_progress = TrainingProgress(
            num_seen_steps_previous_run=self.num_seen_train_steps,
            num_seen_tokens_previous_run=self.global_num_seen_tokens,
            num_seen_steps_current_run=0,
            num_seen_tokens_current_run=0,
            num_target_steps=self.num_target_steps,
            num_target_tokens=self.num_target_tokens,
        )
        checkpointing_callback(training_progress=training_progress)

        num_steps_todo = self.num_target_steps - self.num_seen_train_steps
        num_batches_todo = num_steps_todo * self.gradient_acc_steps
        # Because we might resume training, we add the starting batch id of the data loader
        for _, (micro_batch_id, batch) in zip(range(num_batches_todo), enumerate(train_loader)):
            # Train single batch
            (
                step_performed,
                num_train_steps_done,
                batch_loss,
                gradient_norm_score,
            ) = self._train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fun=loss_fun,
                micro_batch_id=micro_batch_id,
            )
            forward_backward_time_recorder.stop()
            training_progress.num_seen_steps_current_run = num_train_steps_done
            training_progress.num_seen_tokens_current_run = self.global_num_tokens_per_train_step * num_train_steps_done

            # Save the batch loss
            cumulated_losses[0] += batch_loss.item()
            # This works, because we always drop the last batch in case it has less samples than the batch size
            cumulated_losses[-1] += 1  # number of local batches

            # gradient norm is already synced across all ranks
            if gradient_norm_score is not None:
                gradient_norm_scores.append(gradient_norm_score.item())

            batch_length_tensor = torch.tensor(len(batch)).to(device)
            thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)

            self._publish_progress(
                progress_publisher=self.progress_publisher,
                num_train_steps_done=training_progress.num_seen_steps_total,
                dataloader_tag=train_loader.dataloader_tag,
            )
            # Check if model performance should be logged
            if training_progress.num_seen_steps_total % training_log_interval_in_steps == 0 and step_performed:
                forward_backward_time = torch.tensor(forward_backward_time_recorder.delta_t).to(device)
                forward_backward_time_recorder.reset()

                thoughput_aggregator.add_value(
                    key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_time
                )
                synced_num_samples = thoughput_aggregator.get_all_reduced_value(ThroughputAggregationKeys.NUM_SAMPLES)
                synced_forward_backward_time = thoughput_aggregator.get_all_reduced_value(
                    ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
                )
                synced_num_samples_per_second = synced_num_samples / synced_forward_backward_time
                # TODO: insert reducer from outside so Trainer is independent of FSDP
                # add the loss and gradient norm for the LAST batch
                cumulated_losses[1] = batch_loss.item()

                reduced_losses = Reducer.reduce(
                    tensor=cumulated_losses,
                    operation=dist.ReduceOp.SUM,
                    # 1.) summed batch loss / (num batches * world size)
                    # 2.) last batch loss / world size
                    post_processing_fun=lambda t: torch.stack([t[0] / t[-1], t[1] / dist.get_world_size()]),
                )

                train_loss_avg, train_loss_last_batch = (
                    reduced_losses[0],
                    reduced_losses[1],
                )
                losses = {
                    "train loss avg": ResultItem(train_loss_avg, decimal_places=2),
                    "train loss last": ResultItem(train_loss_last_batch, decimal_places=2),
                }

                consumed_tokens = torch.Tensor([training_progress.num_seen_tokens_total])
                metrics = {
                    "consumed tokens": ResultItem(consumed_tokens, 0),
                    "grad norm avg": ResultItem(torch.mean(torch.Tensor(gradient_norm_scores)), 2),
                    "grad norm last": ResultItem(torch.tensor(gradient_norm_scores[-1]), 2),
                }
                gradient_norm_scores = []

                mfu = compute_mfu(
                    synced_num_samples_per_second,
                    sequence_length,
                    theoretical_flops_per_token,
                    theoretical_gpu_peak_performance,
                )
                training_metrics = EvaluationResultBatch(
                    losses=losses,
                    metrics=metrics,
                    # TODO: hardcoded metric key
                    throughput_metrics={
                        "train samples/s": ResultItem(synced_num_samples_per_second, 1),
                        "train mfu": ResultItem(mfu, 2),
                        "lr mean": ResultItem(torch.tensor(scheduler.get_last_lr()).mean()),
                    },
                    dataloader_tag=train_loader.dataloader_tag,
                    num_train_steps_done=training_progress.num_seen_steps_total,
                )
                print_rank_0(training_metrics)
                self._publish_evaluation_result(
                    evaluation_result_publisher=self.evaluation_result_publisher,
                    evaluation_result=training_metrics,
                )
                thoughput_aggregator.remove_keys()

                cumulated_losses = self._reset_tracked_losses()
            if step_performed:
                evaluation_callback(num_train_steps_done=training_progress.num_seen_steps_total)
                checkpointing_callback(training_progress=training_progress)
            # we start the time recoder here again to also capture the time spend loading
            # via the dataloader.
            forward_backward_time_recorder.start()

    def _reset_tracked_losses(self):
        # Initializes and returns a tensor representing the cumulated loss and gradient norm.
        # The tensor is initialized with zeros and its device is set based on the availability of CUDA.

        cumulated_loss_and_gradient_norm = torch.zeros(3)
        if torch.cuda.is_available():
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to(torch.device("cuda"))
        else:
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to("cpu")
        return cumulated_loss_and_gradient_norm

    @staticmethod
    def _publish_progress(
        progress_publisher: MessagePublisher[ProgressUpdate],
        num_train_steps_done: int,
        dataloader_tag: str,
    ):
        # Publishes the progress of the training, i.e., number of training steps done.

        payload = ProgressUpdate(
            num_steps_done=num_train_steps_done,
            experiment_status=ExperimentStatus.TRAIN,
            dataloader_tag=dataloader_tag,
        )
        progress_publisher.publish_message(payload=payload, message_type=MessageTypes.BATCH_PROGRESS_UPDATE)

    @staticmethod
    def _publish_evaluation_result(
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        evaluation_result: EvaluationResultBatch,
    ):
        # Publishes the evaluation result.

        evaluation_result_publisher.publish_message(
            payload=evaluation_result, message_type=MessageTypes.EVALUATION_RESULT
        )
