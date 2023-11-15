from typing import Dict
from llm_gym.logging_broker.messages import (
    BatchProgressUpdate,
    Message,
    ExperimentStatus,
)
from llm_gym.logging_broker.subscriber import MessageSubscriberIF
from rich.progress import (
    Progress,
    MofNCompleteColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.live import Live
from rich.console import Group
from rich.rule import Rule
from rich.text import Text


class DummyProgressSubscriber(MessageSubscriberIF[BatchProgressUpdate]):
    def consume_message(self, message: Message[BatchProgressUpdate]):
        pass


class RichProgressSubscriber(MessageSubscriberIF[BatchProgressUpdate]):
    """A subscriber object for the RichProgress observable."""
    def __init__(
        self,
        num_ranks: int,
        train_split_lengths: Dict[str, int],
        eval_split_lengths: Dict[str, int],
    ) -> None:
        self.num_ranks = num_ranks

        # train split progress bar
        self.train_splits_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        self.train_split_task_ids = {}
        for split_key, split_length in train_split_lengths.items():
            task_id = self.train_splits_progress.add_task(description=split_key, total=split_length)
            self.train_split_task_ids[split_key] = task_id

        # eval split progress bars
        self.eval_splits_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        self.eval_split_task_ids = {}
        for split_key, split_length in eval_split_lengths.items():
            task_id = self.eval_splits_progress.add_task(description=split_key, total=split_length)
            self.eval_split_task_ids[split_key] = task_id

        group = Group(
            Text(text="\n\n\n"),
            Rule(style="#AAAAAA"),
            Text(text="Training", style="blue"),
            self.train_splits_progress,
            Rule(style="#AAAAAA"),
            Text(text="Evaluation", style="blue"),
            self.eval_splits_progress,
        )

        live = Live(group)
        live.start()

    def consume_message(self, message: Message[BatchProgressUpdate]):
        """Consumes a message from a message broker."""
        batch_progress = message.payload

        if batch_progress.experiment_status == ExperimentStatus.TRAIN:
            task_id = self.train_split_task_ids[batch_progress.dataset_tag]
            self.train_splits_progress.update(
                task_id=task_id,
                completed=(batch_progress.train_batch_id + 1) * self.num_ranks,
            )
        else:
            task_id = self.eval_split_task_ids[batch_progress.dataset_tag]
            self.eval_splits_progress.update(
                task_id=task_id,
                completed=(batch_progress.dataset_batch_id + 1) * self.num_ranks,
            )
