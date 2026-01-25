from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, EvaluationResultBatch, ResultItem
from modalities.checkpointing.stateful.app_state import AppState
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import ExperimentStatus, MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import model_predict_batch
from modalities.models.parallelism.pipeline_parallelism import Pipeline
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_parallel_degree
from modalities.running_env.fsdp.reducer import Reducer
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF
from modalities.training.training_progress import TrainingProgress
from modalities.util import TimeRecorder, print_rank_0
from modalities.utils.mfu import MFUCalculatorABC
from modalities.utils.profilers.profilers import SteppableProfilerIF
from modalities.utils.typing_utils import FSDPX


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
        device_mesh: DeviceMesh | None,
        num_seen_train_steps: int,
        global_num_seen_tokens: int,
        num_target_steps: int,
        num_target_tokens: int,
        gradient_clipper: GradientClipperIF,
        profiler: SteppableProfilerIF,
        mfu_calculator: MFUCalculatorABC | None = None,
    ) -> None:
        """
        Initializes the Trainer object.

        Args:
            global_rank (int): The global rank.
            progress_publisher (MessagePublisher[ProgressUpdate]): Progress publisher.
            evaluation_result_publisher (MessagePublisher[EvaluationResultBatch]): Evaluation result publisher.
            gradient_acc_steps (int): Gradient accumulation steps.
            global_num_tokens_per_train_step (int): Global number of tokens per train step.
            dp_degree (int): Data parallelism degree.
            pp_degree (int): Pipeline parallelism degree.
            num_seen_train_steps (int): Number of seen train steps.
            global_num_seen_tokens (int): Global number of seen tokens.
            num_target_steps (int): Number of target steps.
            num_target_tokens (int): Number of target tokens.
            gradient_clipper (GradientClipperIF): Gradient clipper.
            profiler (SteppableProfilerIF): Profiler to profile the training loop.
            mfu_calculator (Optional[MFUCalculatorABC]): MFU calculator.

        Returns:
            None
        """
        self.global_rank = global_rank
        if device_mesh is not None:
            self.dp_degree = get_parallel_degree(
                device_mesh, [ParallelismDegrees.DP_REPLICATE, ParallelismDegrees.DP_SHARD]
            )
            self.pp_degree = get_parallel_degree(device_mesh, [ParallelismDegrees.PP])
        else:  # TODO: we can remove the else part once we refactored out FSDP1
            self.dp_degree = dist.get_world_size()
            self.pp_degree = 1
        self.progress_publisher = progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self.gradient_acc_steps = gradient_acc_steps
        self.global_num_tokens_per_train_step = global_num_tokens_per_train_step
        self.num_seen_train_steps = num_seen_train_steps
        self.num_target_steps = num_target_steps
        self.num_target_tokens = num_target_tokens
        self.global_num_seen_tokens = global_num_seen_tokens
        self.gradient_clipper = gradient_clipper
        self.profiler = profiler
        self.mfu_calculator = mfu_calculator

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
        model_parts: list[FSDPX],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        micro_batch_id: int,
        scheduled_pipeline: Optional[Pipeline] = None,
    ) -> tuple[bool, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Conducts a training step on batch of data.

        Args:
            batch (DatasetBatch): The input batch of data.
            model_parts (list[FSDPX]): The model parts to train.
            optimizer (Optimizer): The optimizer used for training.
            scheduler (LRScheduler): The learning rate scheduler.
            loss_fun (Loss): The loss function used for training.
            micro_batch_id (int): The ID of the micro batch.
            scheduled_pipeline (Optional[Pipeline], optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.

        Returns:
            tuple[bool, int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                A tuple containing the following:
                    - step_performed (bool): Indicates whether a training step was performed.
                    - num_train_steps_done (int): The number of training steps done.
                    - loss (Optional[torch.Tensor]): The computed loss.
                        None, if a non-last stage was processes in pipeline parallelism.
                    - gradient_norm_score (Optional[torch.Tensor]): The gradient norm score,
                        if a training step was performed otherwise return None.
        """
        if scheduled_pipeline is not None:
            pp_schedule = scheduled_pipeline.pp_schedule
            # Pipeline Parallel forward / backward inside step() call
            # with self.train_context(optional_context_parallel_ctx):
            targets, losses = (
                (batch.targets[loss_fun.target_key].contiguous(), [])
                if scheduled_pipeline.has_last_pp_stage
                else (None, None)
            )

            if scheduled_pipeline.has_first_pp_stage:
                pp_schedule.step(batch.samples[model_parts[0].sample_key].contiguous(), target=targets, losses=losses)
            else:
                pp_schedule.step(target=targets, losses=losses)
            loss = (
                torch.mean(torch.stack(losses)).to(losses[0].device) if scheduled_pipeline.has_last_pp_stage else None
            )
        else:
            # else continue with loss calculation
            result_batch = model_predict_batch(model=model_parts[0], batch=batch)
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
        app_state: AppState,
        train_loader: LLMDataLoader,
        loss_fun: Loss,
        training_log_interval_in_steps: int,
        evaluation_callback: Callable[[int], None],
        checkpointing_callback: Callable[[TrainingProgress], None],
        scheduled_pipeline: Pipeline | None = None,
    ):
        """
        Trains the model.

        Args:
            app_state (AppState): The application state containing the model, optimizer and lr scheduler.
            train_loader (LLMDataLoader): The data loader containing the training data.
            loss_fun (Loss): The loss function used for training.
            training_log_interval_in_steps (int): The interval at which training progress is logged.
            evaluation_callback (Callable[[int], None]): A callback function for evaluation.
            checkpointing_callback (Callable[[TrainingProgress], None]): A callback function for checkpointing.
            scheduled_pipeline (Pipeline | None, optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.

        Returns:
            None
        """
        model_parts = app_state.model_parts
        optimizer = app_state.optimizer
        lr_scheduler = app_state.lr_scheduler
        if scheduled_pipeline is None:
            assert len(model_parts) == 1, "Expected a single model part when no scheduled pipeline is provided."
        for m in model_parts:
            m.train()

        local_num_seen_samples = 0
        cumulated_losses = self._reset_tracked_losses()

        # throughput
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # batch loop
        batch: DatasetBatch
        # TODO: why do we need a barrier here?
        # dist.barrier()
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
        with self.profiler as profiler_cm:
            for _, (micro_batch_id, batch) in zip(range(num_batches_todo), enumerate(train_loader)):
                # Train single batch
                (
                    step_performed,
                    num_train_steps_done,
                    batch_loss,
                    gradient_norm_score,
                ) = self._train_batch(
                    batch=batch,
                    model_parts=model_parts,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    loss_fun=loss_fun,
                    micro_batch_id=micro_batch_id,
                    scheduled_pipeline=scheduled_pipeline,
                )
                training_progress.num_seen_steps_current_run = num_train_steps_done
                training_progress.num_seen_tokens_current_run = (
                    self.global_num_tokens_per_train_step * num_train_steps_done
                )

                # The batch_loss might be None if we use pipeline parallelism and are not the last stage.
                if batch_loss is not None:
                    # Save the batch loss
                    cumulated_losses[0] += batch_loss.item()
                    # This works, because we always drop the last batch in case
                    # it has less samples than the batch size
                    cumulated_losses[-1] += 1  # number of local batches

                # gradient norm is already synced across all ranks
                if gradient_norm_score is not None:
                    gradient_norm_scores.append(gradient_norm_score.item())

                local_num_seen_samples += torch.tensor(len(batch)).to(device)

                self._publish_progress(
                    progress_publisher=self.progress_publisher,
                    num_train_steps_done=training_progress.num_seen_steps_total,
                    dataloader_tag=train_loader.dataloader_tag,
                )
                # Check if model performance should be logged
                if training_progress.num_seen_steps_total % training_log_interval_in_steps == 0 and step_performed:
                    forward_backward_time_recorder.stop()
                    forward_backward_time = torch.tensor(forward_backward_time_recorder.delta_t).to(device)
                    forward_backward_time_recorder.reset()
                    forward_backward_time_recorder.start()

                    global_num_seen_samples = local_num_seen_samples * self.dp_degree
                    local_num_seen_samples = 0
                    global_num_samples_per_second = global_num_seen_samples / forward_backward_time

                    # TODO: insert reducer from outside so Trainer is independent of FSDP
                    # add the loss and gradient norm for the LAST batch
                    cumulated_losses[1] = batch_loss.item() if batch_loss is not None else 0.0

                    reduced_losses = Reducer.reduce(
                        tensor=cumulated_losses,
                        operation=dist.ReduceOp.SUM,
                        # 1.) summed batch loss / (num batches * (world size / dp_degree))
                        # 2.) last batch loss / (world size / pp_degree)
                        post_processing_fun=lambda t: torch.stack(
                            [t[0] / t[-1], t[1] / dist.get_world_size() * self.pp_degree]
                        ),
                    )

                    train_loss_avg, train_loss_last_batch = (
                        reduced_losses[0],
                        reduced_losses[1],
                    )
                    losses = {
                        "train loss avg": ResultItem(train_loss_avg, decimal_places=2),
                        "train loss last": ResultItem(train_loss_last_batch, decimal_places=2),
                    }

                    consumed_tokens = torch.tensor(training_progress.num_seen_tokens_total)
                    metrics = {
                        "consumed tokens": ResultItem(consumed_tokens, 0),
                        "grad norm avg": ResultItem(torch.mean(torch.Tensor(gradient_norm_scores)), 2),
                        "grad norm last": ResultItem(torch.tensor(gradient_norm_scores[-1]), 2),
                    }
                    gradient_norm_scores = []
                    mfu_score = torch.tensor(-1.0)
                    if self.mfu_calculator is not None:
                        mfu_score = self.mfu_calculator.compute(num_samples_per_second=global_num_samples_per_second)

                    # Collect peak memory depending on device type. On CPU we fall back to RSS (if available) or -1.
                    if device.type == "cuda":
                        peak_memory_MB = torch.cuda.max_memory_allocated(device) / 1024**2  # in MB
                        torch.cuda.reset_peak_memory_stats(device)
                    else:
                        # ru_maxrss is in kilobytes on Linux; convert to MB. Use -1.0 if resource unavailable.
                        try:
                            import resource  # Standard lib (POSIX). Not available on some platforms.

                            peak_memory_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                        except Exception:
                            peak_memory_MB = -1.0

                    training_metrics = EvaluationResultBatch(
                        losses=losses,
                        metrics=metrics,
                        # TODO: hardcoded metric key
                        throughput_metrics={
                            "train samples/s": ResultItem(global_num_samples_per_second, 1),
                            "train mfu (16-bit)": ResultItem(mfu_score, 2),
                            "lr mean": ResultItem(torch.tensor(lr_scheduler.get_last_lr()).mean()),
                            "peak memory rank 0 (MB)": ResultItem(torch.tensor(peak_memory_MB), 2),
                        },
                        dataloader_tag=train_loader.dataloader_tag,
                        num_train_steps_done=training_progress.num_seen_steps_total,
                    )
                    print_rank_0(f"{datetime.now().isoformat(timespec='seconds')} | {training_metrics}")
                    self._publish_evaluation_result(
                        evaluation_result_publisher=self.evaluation_result_publisher,
                        evaluation_result=training_metrics,
                    )

                    cumulated_losses = self._reset_tracked_losses()
                if step_performed:
                    evaluation_callback(num_train_steps_done=training_progress.num_seen_steps_total)
                    checkpointing_callback(training_progress=training_progress)
                profiler_cm.step()

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
