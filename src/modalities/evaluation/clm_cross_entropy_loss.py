from __future__ import annotations

from enum import Enum
from typing import Dict

import torch

from modalities.batch import InferenceResultBatch
from modalities.evaluation.measure import AggregativeMeasure, AggregativeMeasureFactory
from modalities.loss_functions import CLMCrossEntropyLoss


class LossKeys(Enum):
    CLM_CROSS_ENTROPY = "clm_cross_entropy"
    NUM_SAMPLES = "num_samples"


class AggregativeCLMCrossEntropyLoss(AggregativeMeasure[LossKeys]):
    def __init__(self, target_key: str, prediction_key: str) -> None:
        super().__init__(aggregate_keys=list(LossKeys), reduce_ops={})
        self._loss = CLMCrossEntropyLoss(target_key=target_key, prediction_key=prediction_key, reduction="sum")

    def _postprocess_result_batch(self, batch_result: InferenceResultBatch) -> Dict[LossKeys, torch.Tensor]:
        loss = self._loss(batch_result)
        return {
            LossKeys.CLM_CROSS_ENTROPY: loss,
            LossKeys.NUM_SAMPLES: torch.tensor(len(batch_result)),
        }

    def _calc_measure(self, values: Dict[LossKeys, torch.Tensor]) -> float:
        return values[LossKeys.CLM_CROSS_ENTROPY].item() / values[LossKeys.NUM_SAMPLES].item()


class AggregativeCLMCrossEntropyLossFactory(AggregativeMeasureFactory[LossKeys]):
    def __init__(self, target_key: str, prediction_key: str) -> None:
        self._target_key = target_key
        self._prediction_key = prediction_key

    def create(self) -> AggregativeMeasure:
        return AggregativeCLMCrossEntropyLoss(
            target_key=self._target_key,
            prediction_key=self._prediction_key,
        )
