from pathlib import Path
from typing import Optional

import rich
from rich.console import Group
from rich.panel import Panel

import wandb
from modalities.batch import EvaluationResultBatch
from modalities.logging_broker.messages import Message
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.config.config import AppConfig, WandbConfig

class DummyResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        pass


class RichResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def __init__(self, num_ranks: int) -> None:
        super().__init__()
        self.num_ranks = num_ranks

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        eval_result = message.payload
        losses = {
            f"{eval_result.dataloader_tag} {loss_key}: {loss_values}"
            for loss_key, loss_values in eval_result.losses.items()
        }
        metrics = {
            f"{eval_result.dataloader_tag} {metric_key}: {metric_values}"
            for metric_key, metric_values in eval_result.metrics.items()
        }

        num_samples = (eval_result.global_train_sample_id + 1) * self.num_ranks
        group_content = [f"[yellow]Iteration #{num_samples}:"]
        if losses:
            group_content.append("\nLosses:")
            group_content.extend(losses)
        if metrics:
            group_content.append("\nMetrics:")
            group_content.extend(metrics)
        if losses or metrics:
            rich.print(Panel(Group(*group_content)))


class WandBEvaluationResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    """A subscriber object for the WandBEvaluationResult observable."""

    def __init__(self, num_ranks: int, project: str, experiment_id: str, mode: WandbConfig.WandbMode, dir: Path, 
                 experiment_config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self.num_ranks = num_ranks

        # experiment_config_json = None
        # if experiment_config is not None:
        #     experiment_config_json = experiment_config.model_dump(mode="json")

        wandb.init(project=project, name=experiment_id, mode=mode.value.lower(), dir=dir, config=experiment_config)

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        eval_result = message.payload
        losses = {
            f"{eval_result.dataloader_tag} {loss_key}": loss_values
            for loss_key, loss_values in eval_result.losses.items()
        }
        metrics = {
            f"{eval_result.dataloader_tag} {metric_key}": metric_values
            for metric_key, metric_values in eval_result.metrics.items()
        }
        # TODO step is not semantically correct here. Need to check if we can rename step to num_samples
        wandb.log(
            data=losses, step=eval_result.global_train_sample_id + 1
        )  # (eval_result.train_local_sample_id + 1) * self.num_ranks)
        wandb.log(
            data=metrics, step=eval_result.global_train_sample_id + 1
        )  # (eval_result.train_local_sample_id + 1) * self.num_ranks)
        throughput_metrics = {
            f"{eval_result.dataloader_tag} {metric_key}": metric_values
            for metric_key, metric_values in eval_result.throughput_metrics.items()
        }
        wandb.log(
            data=throughput_metrics, step=eval_result.global_train_sample_id + 1
        ) 
