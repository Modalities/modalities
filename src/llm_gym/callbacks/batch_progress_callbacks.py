from abc import abstractmethod
from typing import Any, Dict
import torch.distributed as dist
from rich.progress import Progress


class ProgressCallbackIF:

    @abstractmethod
    def __call__(self, epoch_increment: int = 0, batch_increment: int = 1, split_key: str = None) -> Any:
        raise NotImplementedError


class DummyProgressCallback(ProgressCallbackIF):

    def __call__(self, epoch_increment: int = 0, batch_increment: int = 1, split_key: str = None) -> Any:
        pass


class PrintProgressCallback(ProgressCallbackIF):
    def __init__(self, num_epochs: int, split_lengths: Dict[str, int], print_frequency: int = 0.1, subscribing_global_rank: int = None) -> None:
        self.subscribing_global_rank = subscribing_global_rank
        if self.subscribing_global_rank is not None and dist.get_rank() == self.subscribing_global_rank:
            self.print_frequency = print_frequency
            self.split_keys = list(split_lengths.keys())
            self.split_progress_status = {split_key: {"target": split_length, "progress": 0, "last_print": -1}
                                          for split_key, split_length in split_lengths.items()}
            self.epoch_progress_status = {"target": num_epochs, "progress": 0}

    def __call__(self, epoch_increment: int = 0, batch_increment: int = 1, split_key: str = None) -> Any:
        if self.subscribing_global_rank is not None and dist.get_rank() == self.subscribing_global_rank:
            if epoch_increment > 0:
                PrintProgressCallback._print_impl(self.split_progress_status, self.epoch_progress_status)
                self.epoch_progress_status["progress"] += epoch_increment
                for sk in self.split_keys:
                    self.split_progress_status[sk]["progress"] = 0
                    self.split_progress_status[sk]["last_print"] = -1
                PrintProgressCallback._print_impl(self.split_progress_status, self.epoch_progress_status)

            else:
                self.split_progress_status[split_key]["progress"] += batch_increment
                PrintProgressCallback._print_update(self.split_progress_status, self.epoch_progress_status,
                                                    print_frequency=self.print_frequency, split_key=split_key)

    @staticmethod
    def _print_update(split_progress_status: Dict, epoch_progress_status: Dict, print_frequency: int, split_key: str):
        last_print = split_progress_status[split_key]["last_print"]
        target = split_progress_status[split_key]["target"]

        progress = split_progress_status[split_key]["progress"]

        if last_print//(target*print_frequency) < progress//(target*print_frequency):
            split_progress_status[split_key]["last_print"] = progress
            PrintProgressCallback._print_impl(split_progress_status, epoch_progress_status)

    @staticmethod
    def _print_impl(split_progress_status: Dict, epoch_progress_status: Dict):
        status_string = "\n\n===============================================================================================\n"
        status_string += f"epoch: {epoch_progress_status['progress']} / {epoch_progress_status['target']} \n"
        for split_key, split_progress in split_progress_status.items():
            if split_key != "epochs":
                status_string += f"\t split {split_key}: {split_progress['progress']} / {split_progress['target']} \n"
        print(status_string)


class RichProgressCallback:
    def __init__(self, subscribing_global_rank: int, num_epochs: int, split_lengths: Dict[str, int]) -> None:
        self.subscribing_global_rank = subscribing_global_rank
        if self.subscribing_global_rank is not None and dist.get_rank() == self.subscribing_global_rank:
            self.progress = Progress()
            self.split_keys = list(split_lengths.keys())
            self.tasks = {}
            for split_key, split_length in split_lengths.items():
                task = self.progress.add_task(description=split_key, total=split_length)
                self.tasks[split_key] = task
            task = self.progress.add_task(description="epochs", total=num_epochs)
            self.tasks["epochs"] = task

    def __call__(self, epoch_increment: int = 0, batch_increment: int = 1, split_key: str = None) -> Any:
        if self.subscribing_global_rank is not None and dist.get_rank() == self.subscribing_global_rank:
            if epoch_increment > 0:
                self.progress.update(self.tasks["epochs"], advance=epoch_increment)
                for sk in self.split_keys:
                    self.progress.update(self.tasks[sk], completed=0)
            else:
                self.progress.update(self.tasks[split_key], advance=batch_increment)
