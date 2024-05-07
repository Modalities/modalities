from pathlib import Path

import rich
import torch
from rich.console import Group
from rich.panel import Panel

import wandb
from modalities.batch import EvaluationResultBatch
from modalities.config.config import WandbMode
from modalities.logging_broker.message_broker import MessageBroker
from modalities.logging_broker.messages import BatchProgressUpdate, Message, MessageTypes, ModelState
from modalities.logging_broker.subscriber import MessageSubscriberIF


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

        num_samples = (eval_result.train_step_id + 1) * self.num_ranks
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

    def __init__(
        self,
        project: str,
        experiment_id: str,
        mode: WandbMode,
        logging_directory: Path,
        config_file_path: Path,
    ) -> None:
        super().__init__()

        run = wandb.init(project=project, name=experiment_id, mode=mode.value.lower(), dir=logging_directory)
        self.last_batch_progress_update: BatchProgressUpdate = None

        run.log_artifact(config_file_path, name=f"config_{wandb.run.id}", type="config")

    def consume_message(self, message: Message):
        """Consumes a message from a message broker."""
        if message.message_type == MessageTypes.EVALUATION_RESULT:
            self._consum_evaluation_results_batch(message)
        elif message.message_type == MessageTypes.BATCH_PROGRESS_UPDATE:
            self.last_batch_progress_update = message.payload
        elif message.message_type == MessageTypes.MODEL_STATE:
            self._consum_model_state(message)

        # wandb.log({"tokens_loss": wandb.plot.scatter("num_tokens", "loss", title="Tokens vs Loss")})
        # wandb.log({"steps_loss": wandb.plot.scatter("steps_loss", "loss", title="Steps vs Loss")})
        # wandb.log({"samples_loss": wandb.plot.scatter("samples_loss", "loss", title="Samples vs Loss")})

    def _consum_model_state(self, message: Message[ModelState]):
        wandb.log(data={message.payload.key: message.payload.value}, step=self.last_batch_progress_update.step_id + 1)

    def _consum_evaluation_results_batch(self, message: Message[EvaluationResultBatch]):
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
            data=losses, step=eval_result.train_step_id + 1
        )  # (eval_result.train_local_sample_id + 1) * self.num_ranks)
        wandb.log(
            data=metrics, step=eval_result.train_step_id + 1
        )  # (eval_result.train_local_sample_id + 1) * self.num_ranks)
        throughput_metrics = {
            f"{eval_result.dataloader_tag} {metric_key}": metric_values
            for metric_key, metric_values in eval_result.throughput_metrics.items()
        }

        wandb.log(data=throughput_metrics, step=eval_result.train_step_id + 1)


class ModelStatePublisher:
    def __init__(self, message_broker: MessageBroker):
        self.message_broker = message_broker

    def log_activations(self, module, input, output):
        entropy = torch.distributions.Categorical(probs=output).entropy()
        payload = ModelState(key=ModelState.KeyEnum.ACTIVATION_ENTROPY, value=entropy.item())
        message = Message(payload=payload, message_type=MessageTypes.MODEL_STATE)
        self.message_broker.distribute_message(message)

    def log_attention_scores(self, module, input, output):
        # TODO: Implement logging of attention scores using WandBEvaluationResultSubscriber
        pass
