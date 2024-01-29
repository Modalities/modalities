from __future__ import annotations

from enum import Enum
from typing import Dict

import torch

from modalities.batch import InferenceResultBatch
from modalities.evaluation.measure import AggregativeMeasure, AggregativeMeasureFactory
from modalities.loss_functions import CLMCrossEntropyLoss


class PerplexityKeys(Enum):
    PERPLEXITY = "perplexity"
    NUM_SAMPLES = "num_samples"


class AggregativePerplexity(AggregativeMeasure[PerplexityKeys]):
    def __init__(self, target_key: str, prediction_key: str) -> None:
        super().__init__(aggregate_keys=[PerplexityKeys.PERPLEXITY], reduce_ops={})
        self._loss = CLMCrossEntropyLoss(target_key=target_key, prediction_key=prediction_key, reduction="none")

    def _postprocess_result_batch(self, batch_result: InferenceResultBatch) -> Dict[PerplexityKeys, torch.Tensor]:
        loss = self._loss(batch_result)
        return {
            PerplexityKeys.PERPLEXITY: torch.exp(loss).sum(),
            PerplexityKeys.NUM_SAMPLES: torch.tensor(len(batch_result)),
        }

    def _calc_measure(self, values: Dict[PerplexityKeys, torch.Tensor]) -> float:
        return values[PerplexityKeys.PERPLEXITY].item() / values[PerplexityKeys.NUM_SAMPLES].item()


class AggregativePerplexityFactory(AggregativeMeasureFactory[PerplexityKeys]):
    def __init__(self, target_key: str, prediction_key: str) -> None:
        self._target_key = target_key
        self._prediction_key = prediction_key

    def create(self) -> AggregativeMeasure[PerplexityKeys]:
        return AggregativePerplexity(
            target_key=self._target_key,
            prediction_key=self._prediction_key,
        )
