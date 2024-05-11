from pydantic import BaseModel

from modalities.messaging.evaluation.processors.processors import LocalProcessorIF
from modalities.messaging.evaluation.states import IntervalState
from modalities.messaging.messages.payloads import BatchProgressUpdate


class BatchProgressUpdateProcessor(LocalProcessorIF[BatchProgressUpdate]):
    def __init__(self):
        pass

    def process(self, payload: BatchProgressUpdate, current_local_step_state: IntervalState):
        current_local_step_state.meta_information = payload


class BatchProgressUpdateProcessorConfig(BaseModel):
    pass
