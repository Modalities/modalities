import rich
import wandb
from rich.console import Group
from rich.panel import Panel

from llm_gym.batch import EvaluationResultBatch
from llm_gym.logging_broker.messages import Message
from llm_gym.logging_broker.subscriber import MessageSubscriberIF


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

        step = (eval_result.train_batch_id + 1) * self.num_ranks
        group_content = [f"[yellow]Iteration #{step}:"]
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

    def __init__(self, num_ranks: int, project: str, experiment_id: str) -> None:
        super().__init__()
        self.num_ranks = num_ranks
        wandb.init(project=project, name=experiment_id)

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
        wandb.log(data=losses, step=(eval_result.train_batch_id + 1) * self.num_ranks)
        wandb.log(data=metrics, step=(eval_result.train_batch_id + 1) * self.num_ranks)
