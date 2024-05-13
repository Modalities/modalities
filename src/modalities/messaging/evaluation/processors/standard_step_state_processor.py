from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

from modalities.messaging.evaluation.processors.processors import GlobalProcessorIF, LocalProcessorIF
from modalities.messaging.evaluation.states import IntervalState, LocalReduceOperations, RankReduceOperations, Trackable
from modalities.messaging.messages.payloads import EvaluationResult, StepState


class TrackablesKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"
    NUM_STEPS = "NUM_STEPS"
    CUMM_BATCH_LOSS = "CUMM_BATCH_LOSS"
    LAST_BATCH_LOSS = "LAST_BATCH_LOSS"

    # only in train
    CUMM_GRADIENT_NORM = "CUMM_GRADIENT_NORM"
    LAST_BATCH_GRADIENT_NORM = "LAST_BATCH_GRADIENT_NORM"
    LAST_SCHEDULER_LR = "LAST_SCHEDULER_LR"


class StandardLocalStepStateProcessor(LocalProcessorIF[StepState]):
    def __init__(self):
        self.trackable_key_to_reduce_op = {
            TrackablesKeys.NUM_SAMPLES: (LocalReduceOperations.SUM, RankReduceOperations.SUM),
            TrackablesKeys.FORWARD_BACKWARD_TIME: (LocalReduceOperations.SUM, RankReduceOperations.MAX),
            TrackablesKeys.NUM_STEPS: (LocalReduceOperations.SUM, RankReduceOperations.NONE),
            TrackablesKeys.CUMM_BATCH_LOSS: (LocalReduceOperations.SUM, RankReduceOperations.SUM),
            TrackablesKeys.LAST_BATCH_LOSS: (LocalReduceOperations.REPLACE, RankReduceOperations.SUM),
            TrackablesKeys.CUMM_GRADIENT_NORM: (LocalReduceOperations.SUM, RankReduceOperations.NONE),
            TrackablesKeys.LAST_BATCH_GRADIENT_NORM: (LocalReduceOperations.REPLACE, RankReduceOperations.NONE),
            TrackablesKeys.LAST_SCHEDULER_LR: (LocalReduceOperations.REPLACE, RankReduceOperations.NONE),
        }

    def process(self, payload: StepState, current_local_step_state: IntervalState):
        for key in TrackablesKeys:
            # during evaluation we don't have gradient norm tracking,
            # that's why we need this check here.
            if key in payload.trackable_values:
                trackable = Trackable(
                    key=key,
                    tag="",
                    value=payload.trackable_values[key],
                    local_reduce_op=self.trackable_key_to_reduce_op[key][0],
                    rank_reduce_op=self.trackable_key_to_reduce_op[key][1],
                )
                current_local_step_state.trackables.set_trackable(trackable)


class StandardGlobalStepStateProcessor(GlobalProcessorIF):
    def __init__(self, world_size: int):
        self.world_size = world_size

    def process(self, interval_state: IntervalState, eval_result: EvaluationResult):
        trackables = interval_state.trackables
        # throughput
        num_samples = trackables.get_trackable[TrackablesKeys.NUM_SAMPLES]
        forward_backward_time = trackables.get_trackable[TrackablesKeys.FORWARD_BACKWARD_TIME]
        throughput = num_samples / forward_backward_time
        eval_result.trackables[
            f"{interval_state.meta_information.experiment_status} throughput [samples/s]"
        ] = throughput

        # losses
        avg_loss = trackables.get_trackable(TrackablesKeys.CUMM_BATCH_LOSS) / num_samples
        last_batch_loss = trackables.get_trackable(TrackablesKeys.LAST_BATCH_LOSS) / self.world_size
        eval_result.trackables["avg loss"] = avg_loss
        eval_result.trackables["last batch loss"] = last_batch_loss

        # gradient norm
        if TrackablesKeys.CUMM_GRADIENT_NORM in trackables.state:
            avg_gradient_norm = trackables.get_keys(TrackablesKeys.CUMM_GRADIENT_NORM) / trackables.get_trackable(
                TrackablesKeys.NUM_STEPS
            )
            eval_result.trackables["avg gradient norm"] = avg_gradient_norm

        if TrackablesKeys.LAST_BATCH_GRADIENT_NORM in trackables.state:
            eval_result.trackables["last batch gradient norm"] = trackables.get_keys(
                TrackablesKeys.LAST_BATCH_GRADIENT_NORM
            )

        return eval_result


class StandardGlobalStepStateProcessorConfig(BaseModel):
    world_size: Annotated[int, Field(strict=True, ge=1)]


class StandardLocalStepStateProcessorConfig(BaseModel):
    pass
