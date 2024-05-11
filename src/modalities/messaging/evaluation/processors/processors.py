from typing import Generic, TypeVar

from modalities.messaging.evaluation.states import IntervalState
from modalities.messaging.messages.payloads import EvaluationResult

T = TypeVar("T")


class LocalProcessorIF(Generic[T]):
    def process(self, payload: T, current_local_step_state: IntervalState):
        raise NotImplementedError


class GlobalProcessorIF:
    def process(self, interval_state: IntervalState, eval_result: EvaluationResult):
        raise NotImplementedError
