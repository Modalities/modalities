import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import rich
import torch
import wandb
import yaml
from rich.console import Group
from rich.panel import Panel

from modalities.batch import EvaluationResultBatch
from modalities.config.config import WandbMode
from modalities.logging_broker.messages import Message
from modalities.logging_broker.subscriber import MessageSubscriberIF


class DummyResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        pass

    def consume_dict(self, message_dict: dict[str, Any]):
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

        num_samples = eval_result.num_train_steps_done * self.num_ranks
        group_content = [f"[yellow]Iteration #{num_samples}:"]
        if losses:
            group_content.append("\nLosses:")
            group_content.extend(losses)
        if metrics:
            group_content.append("\nMetrics:")
            group_content.extend(metrics)
        if losses or metrics:
            rich.print(Panel(Group(*group_content)))

    def consume_dict(self, message_dict: dict[str, Any]):
        raise NotImplementedError


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

        with open(config_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.run = wandb.init(
            project=project, name=experiment_id, mode=mode.value.lower(), dir=logging_directory, config=config
        )

        self.run.log_artifact(config_file_path, name=f"config_{wandb.run.id}", type="config")

    def consume_dict(self, message_dict: dict[str, Any]):
        for k, v in message_dict.items():
            self.run.config[k] = v

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        eval_result = message.payload

        losses = {
            f"{eval_result.dataloader_tag} {loss_key}": loss_values.value
            for loss_key, loss_values in eval_result.losses.items()
        }
        metrics = {
            f"{eval_result.dataloader_tag} {metric_key}": metric_values.value
            for metric_key, metric_values in eval_result.metrics.items()
        }
        # TODO step is not semantically correct here. Need to check if we can rename step to num_samples
        wandb.log(
            data=losses, step=eval_result.num_train_steps_done
        )  # (eval_result.train_local_sample_id + 1) * self.num_ranks)
        wandb.log(
            data=metrics, step=eval_result.num_train_steps_done
        )  # (eval_result.train_local_sample_id + 1) * self.num_ranks)
        throughput_metrics = {
            f"{eval_result.dataloader_tag} {metric_key}": metric_values.value
            for metric_key, metric_values in eval_result.throughput_metrics.items()
        }

        wandb.log(data=throughput_metrics, step=eval_result.num_train_steps_done)

        # wandb.log({"tokens_loss": wandb.plot.scatter("num_tokens", "loss", title="Tokens vs Loss")})
        # wandb.log({"steps_loss": wandb.plot.scatter("steps_loss", "loss", title="Steps vs Loss")})
        # wandb.log({"samples_loss": wandb.plot.scatter("samples_loss", "loss", title="Samples vs Loss")})


class EvaluationResultToDiscSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    """A subscriber that writes EvaluationResultBatch messages to a JSONL file."""

    def __init__(self, output_file_path: Path) -> None:
        super().__init__()
        self.output_file_path = output_file_path

    def consume_dict(self, message_dict: dict[str, Any]):
        """Optional: log config data if needed (here: no-op)."""
        pass

    @staticmethod
    def _convert_evaluation_result_batch(eval_result_batch: EvaluationResultBatch) -> dict[str, Any]:
        """
        Recursively convert EvaluationResultBatch structure to JSON-serializable format.
        Handles dataclasses and torch.Tensor.
        """
        if is_dataclass(eval_result_batch):
            result_dict = {}
            for k, v in asdict(eval_result_batch).items():
                result_dict[k] = EvaluationResultToDiscSubscriber._convert_evaluation_result_batch(v)
            return result_dict

        elif isinstance(eval_result_batch, dict):
            return {
                k: EvaluationResultToDiscSubscriber._convert_evaluation_result_batch(v)
                for k, v in eval_result_batch.items()
            }
        elif isinstance(eval_result_batch, list):
            return [EvaluationResultToDiscSubscriber._convert_evaluation_result_batch(v) for v in eval_result_batch]
        elif isinstance(eval_result_batch, torch.Tensor):
            return eval_result_batch.item() if eval_result_batch.ndim == 0 else eval_result_batch.tolist()
        else:
            return eval_result_batch

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Writes the evaluation result to the JSONL file if rank 0."""
        if torch.distributed.get_rank() == 0:
            eval_result = message.payload
            # Convert the dataclass (including nested dataclasses) to a dictionary
            record_converted = EvaluationResultToDiscSubscriber._convert_evaluation_result_batch(eval_result)
            with self.output_file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record_converted) + "\n")
