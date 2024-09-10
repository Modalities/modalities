from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingProgress:
    """
    Dataclass to store the training progress.

    Attributes:

        num_seen_steps_current_run (int): Number of seen steps in the current run.
        num_seen_tokens_current_run (int): Number of seen tokens in the current run.
        num_target_steps (int): Target number of steps.
        num_target_tokens (int): Target number of tokens.
        num_seen_steps_previous_run (Optional[int]): Number of seen steps in the previous run.
        num_seen_tokens_previous_run (Optional[int]): Number of seen tokens in the previous run.
    """

    num_seen_steps_current_run: int
    num_seen_tokens_current_run: int
    num_target_steps: int
    num_target_tokens: int
    num_seen_steps_previous_run: Optional[int] = 0
    num_seen_tokens_previous_run: Optional[int] = 0

    @property
    def num_seen_steps_total(self) -> int:
        return self.num_seen_steps_current_run + self.num_seen_steps_previous_run

    @property
    def num_seen_tokens_total(self) -> int:
        return self.num_seen_tokens_current_run + self.num_seen_tokens_previous_run
