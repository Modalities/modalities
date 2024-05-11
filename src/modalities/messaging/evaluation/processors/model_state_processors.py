from enum import Enum
from typing import List

import torch
from pydantic import BaseModel

from modalities.messaging.evaluation.processors.processors import GlobalProcessorIF, LocalProcessorIF
from modalities.messaging.evaluation.processors.standard_step_state_processor import (
    TrackablesKeys as StandardTrackablesKeys,
)
from modalities.messaging.evaluation.states import IntervalState, LocalReduceOperations, RankReduceOperations, Trackable
from modalities.messaging.messages.payloads import EvaluationResult, ModelState


class TrackablesKeys(Enum):
    CROSS_ENTROPY = "CROSS_ENTROPY"


class LocalModelStateProcessor(LocalProcessorIF[ModelState]):
    def __init__(self, trackable_keys: List[TrackablesKeys]):
        self.trackable_keys = trackable_keys
        self.trackable_key_to_process_fun = {TrackablesKeys.CROSS_ENTROPY: self._process_cross_entropy}

    def process(self, payload: ModelState, current_local_step_state: IntervalState):
        for key in self.trackable_keys:
            self.trackable_key_to_process_fun[key](payload, current_local_step_state)

    def _process_cross_entropy(self, payload: ModelState, current_local_step_state: IntervalState):
        entropy_score = torch.distributions.Categorical(logits=payload.value).entropy()
        layer_name = payload.key
        trackable = Trackable(
            key=TrackablesKeys.CROSS_ENTROPY,
            tag=layer_name,
            value=entropy_score,
            local_reduce_op=LocalReduceOperations.SUM,
            rank_reduce_op=RankReduceOperations.SUM,
        )
        current_local_step_state.trackables.set_trackable(trackable)


class GlobalModelStateProcessor(GlobalProcessorIF):
    def __init__(self):
        pass

    def process(self, interval_state: IntervalState, eval_result: EvaluationResult):
        trackables = interval_state.trackables

        # gradient norm
        if TrackablesKeys.CROSS_ENTROPY in trackables.state:
            ce_trackable = trackables.get_trackable(TrackablesKeys.CROSS_ENTROPY)
            avg_cross_entropy = trackables.get_trackable(TrackablesKeys.CROSS_ENTROPY) / trackables.get_trackable(
                StandardTrackablesKeys.NUM_SAMPLES
            )
            eval_result.trackables[f"{ce_trackable.tag} avg cross entropy"] = avg_cross_entropy


class LocalModelStateProcessorConfig(BaseModel):
    trackable_keys: List[TrackablesKeys]


class GlobalModelStateProcessorConfig(BaseModel):
    pass
