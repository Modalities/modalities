from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import torch.nn as nn

from modalities.batch import InferenceResultBatch


class ExperimentStatus(Enum):
    TRAIN = "TRAIN"
    EVALUATION = "EVALUATION"


@dataclass
class BatchProgressUpdate:
    """Object holding the state of the current batch computation progress."""

    step_id: int
    # Note: in case of ExperimentState.TRAIN, dataset_batch_id=global_train_batch_id
    experiment_status: ExperimentStatus
    dataloader_tag: str


@dataclass
class ModelState:
    class ModelStateKeys(Enum):
        ACTIVATION_ENTROPY = "ACTIVATION_ENTROPY"
        SELF_ATTENTION_ENTROPY = "SELF_ATTENTION_ENTROPY"

    key: ModelStateKeys
    value: float


@dataclass
class TrainStepState:
    @dataclass
    class Trackables:
        loss: float
        gradient_norm_score: Optional[float] = None
        # learning rate of the first parameter according to the lr scheduler
        scheduler_lr_first: Optional[float] = None
        num_samples: int
        forward_backward_time: float

    @dataclass
    class MetaInformation:
        step_id: int
        dataloader_tag: str
        loss_fun_tag: str
        experiment_status: ExperimentStatus

    trackables: Trackables
    inference_result_batch: InferenceResultBatch
    meta_information: MetaInformation
    model_parameters: Iterator[nn.Parameter]


@dataclass
class EvalStepState:
    @dataclass
    class Trackables:
        num_samples: int
        forward_backward_time: float

    @dataclass
    class MetaInformation:
        step_id: int
        num_steps: int
        dataloader_tag: str
        experiment_status: ExperimentStatus

    trackables: Trackables
    inference_result_batch: InferenceResultBatch
    meta_information: MetaInformation
