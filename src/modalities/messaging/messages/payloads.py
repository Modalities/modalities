from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

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
    module_alias: str
    module_input: Tuple[torch.Tensor]
    module_output: torch.Tensor
    module: nn.Module


@dataclass
class StepState:
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
    experiment_status: ExperimentStatus
    losses: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())
    metrics: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())
    trackables: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())
