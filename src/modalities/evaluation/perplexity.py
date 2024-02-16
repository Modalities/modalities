from __future__ import annotations

from enum import Enum
from typing import Dict

import torch
import torch.distributed as dist

from modalities.batch import InferenceResultBatch
from modalities.evaluation.measure import AggregativeMeasure, AggregativeMeasureFactory
from modalities.loss_functions import CLMCrossEntropyLoss


class PerplexityKeys(Enum):
    PERPLEXITY = "loss"
    NUM_SAMPLES = "num_samples"


class AggregativePerplexity(AggregativeMeasure[PerplexityKeys]):
    def __init__(self, target_key: str, prediction_key: str, local_rank: int) -> None:
        super().__init__(
            aggregate_keys=list(PerplexityKeys),
            reduce_ops={k: dist.ReduceOp.SUM for k in PerplexityKeys},
            tag="Perplexity",
            local_rank=local_rank,
        )
        self._target_key = target_key
        self._loss = CLMCrossEntropyLoss(target_key=target_key, prediction_key=prediction_key, reduction="none")

    def _postprocess_result_batch(self, batch_result: InferenceResultBatch) -> Dict[PerplexityKeys, torch.Tensor]:
        loss = self._loss(batch_result)  # shape: (batch_size * seq_len)
        batch_size, seq_len = batch_result.get_targets(self._target_key).shape
        loss = loss.view(batch_size, seq_len)  # shape: (batch_size, seq_len)
        perplexity = torch.exp(loss.sum(-1) / seq_len)
        return {
            PerplexityKeys.PERPLEXITY: perplexity.sum(),
            PerplexityKeys.NUM_SAMPLES: torch.tensor(len(batch_result)),
        }

    def _calc_measure(self, values: Dict[PerplexityKeys, torch.Tensor]) -> torch.Tensor:
        return values[PerplexityKeys.PERPLEXITY] / values[PerplexityKeys.NUM_SAMPLES]


class AggregativePerplexityFactory(AggregativeMeasureFactory[PerplexityKeys]):
    def __init__(self, target_key: str, prediction_key: str) -> None:
        self._target_key = target_key
        self._prediction_key = prediction_key

    def create(self, local_rank: int) -> AggregativeMeasure[PerplexityKeys]:
        return AggregativePerplexity(
            target_key=self._target_key,
            prediction_key=self._prediction_key,
            local_rank=local_rank,
        )
