from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator

import torch
import torch.nn as nn

from modalities.batch import InferenceResultBatch


class ExperimentStatus(Enum):
    TRAIN = "TRAIN"
    EVALUATION = "EVALUATION"


@dataclass
class BatchProgressUpdate:
    """Object holding the state of the current batch computation progress."""

    step_id: int
    num_steps: int
    # Note: in case of ExperimentState.TRAIN, dataset_batch_id=global_train_batch_id
    experiment_status: ExperimentStatus
    dataloader_tag: str


@dataclass
class ModelState:
    tensor_key: str
    value: Iterator[nn.Parameter]


@dataclass
class StepState:
    # @dataclass
    # class TrackableValues:
    # loss: float
    # num_samples: int
    # forward_backward_time: float
    # gradient_norm_score: Optional[float] = None
    # # learning rate of the first parameter according to the lr scheduler
    # scheduler_lr_first: Optional[float] = None

    @dataclass
    class MetaInformation:
        step_id: int
        num_steps: int
        dataloader_tag: str
        loss_fun_tag: str
        experiment_status: ExperimentStatus

    trackable_values: Dict[Enum, float | int | torch.Tensor]
    inference_result_batch: InferenceResultBatch
    meta_information: MetaInformation


@dataclass
class EvaluationResult:
    """Data class for storing the results of a single or multiple batches.
    Also entire epoch results are stored in here."""

    dataloader_tag: str
    train_step_id: int
    losses: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())
    metrics: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())
    trackables: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())
