from enum import Enum
from typing import Generic, TypeVar

from modalities.messaging.evaluation.states import IntervalState, LocalReduceOperations, RankReduceOperations, Trackable
from modalities.messaging.messages.payloads import BatchProgressUpdate, EvaluationResult, StepState


class TrackablesKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"
    NUM_STEPS = "NUM_STEPS"
    CUMM_BATCH_LOSS = "CUMM_BATCH_LOSS"
    LAST_BATCH_LOSS = "LAST_BATCH_LOSS"
    # only in train
    CUMM_GRADIENT_NORM = "CUMM_GRADIENT_NORM"
    LAST_BATCH_GRADIENT_NORM = "LAST_BATCH_GRADIENT_NORM"


T = TypeVar("T")


class LocalProcessorIF(Generic[T]):
    def process(self, payload: T, current_local_step_state: IntervalState):
        raise NotImplementedError


class GlobalProcessorIF:
    def process(self, current_step_state: IntervalState, eval_result: EvaluationResult):
        raise NotImplementedError


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
        }

    def process(self, payload: StepState, current_local_step_state: IntervalState):
        for key in TrackablesKeys:
            # during evaluation we don't have gradient norm tracking,
            # that's why we need this check here.
            if key in payload.trackable_values:
                trackable = Trackable(
                    key=key,
                    value=payload.trackable_values[key],
                    local_reduce_op=self.trackable_key_to_reduce_op[key][0],
                    rank_reduce_op=self.trackable_key_to_reduce_op[key][1],
                )
                current_local_step_state.trackables.set_trackable(trackable)


class StandardGlobalStepStateProcessor(GlobalProcessorIF):
    def __init__(self, world_size: int):
        self.world_size = world_size

    def process(self, current_step_state: IntervalState, eval_result: EvaluationResult):
        trackables = current_step_state.trackables
        # throughput
        num_samples = trackables.get_trackable[TrackablesKeys.NUM_SAMPLES]
        forward_backward_time = trackables.get_trackable[TrackablesKeys.FORWARD_BACKWARD_TIME]
        num_samples / forward_backward_time

        # losses
        trackables.get_trackable(TrackablesKeys.CUMM_BATCH_LOSS) / trackables.get_trackable(TrackablesKeys.NUM_STEPS)

        trackables.get_trackable(TrackablesKeys.LAST_BATCH_LOSS) / self.world_size

        # gradient norm
        if TrackablesKeys.CUMM_GRADIENT_NORM in trackables.state:
            trackables.get_keys(TrackablesKeys.CUMM_GRADIENT_NORM)

        [
            payload.trackables.gradient_norm_score
            for payload in self.train_step_state_history
            if payload.trackables.gradient_norm_score is not None
        ]

        # if len(gradient_norm_scores) > 0:
        #     metrics = {
        #         "grad_norm_avg": np.mean(gradient_norm_scores),
        #         "grad_norm_last_batch": gradient_norm_scores[-1],
        #     }
        # else:
        #     metrics = {}

        # result_batch = EvaluationResult(
        #     losses=losses,
        #     metrics=metrics,
        #     # TODO: hardcoded metric key
        #     meta_metrics={
        #         "training_synced_num_samples_per_second": synced_num_samples_per_second,
        #         "scheduler_lr_first": last_train_step_state.trackables.scheduler_lr_first,
        #     },
        #     dataloader_tag=last_train_step_state.meta_information.dataloader_tag,
        #     train_step_id=last_train_step_state.meta_information.step_id,
        # )
        # return result_batch


class BatchProgressUpdateProcessor(LocalProcessorIF[BatchProgressUpdate]):
    def __init__(self):
        pass

    def process(self, payload: BatchProgressUpdate, current_local_step_state: IntervalState):
        current_local_step_state.meta_information = payload
