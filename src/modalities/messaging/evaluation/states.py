from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

from modalities.messaging.messages.payloads import ExperimentStatus
from modalities.running_env.fsdp.reducer import Reducer


class RankReduceOperations(dist.ReduceOp):
    NONE = "NONE"


class LocalReduceOperations(Enum):
    @staticmethod
    def _sum(last, current) -> float | int:
        return last + current

    @staticmethod
    def _max(last, current) -> float | int:
        return np.max(last, current)

    @staticmethod
    def _replace(_, current) -> float | int:
        return current

    SUM = _sum
    MAX = _max
    REPLACE = _replace


@dataclass
class Trackable:
    key: Enum
    value: float | int | torch.Tensor
    rank_reduce_op: RankReduceOperations
    local_reduce_op: LocalReduceOperations


@dataclass
class IntervalState:
    class TrackableCollection:
        def __init__(self):
            self.state: Dict[Enum, Trackable] = {}

        def set_trackable(self, trackable: Trackable):
            if trackable.key in self.state:
                local_reduce_op_fun = trackable.local_reduce_op.value
                self.state[trackable.key] = local_reduce_op_fun([self.state[trackable.key], trackable.value])
            else:
                self.state[trackable.key] = trackable

        def get_trackable(self, key: Enum) -> Trackable:
            return self.state[key]

        def get_keys(self) -> List[Enum]:
            return list(self.state.keys())

    @dataclass
    class MetaInformation:
        step_id: int
        num_steps: int
        dataloader_tag: str
        experiment_status: ExperimentStatus

    trackables: TrackableCollection = TrackableCollection()
    meta_information: MetaInformation

    def reduce_values_across_ranks(self) -> Dict[Enum, torch.Tensor]:
        reduce_op_to_trackables: Dict[RankReduceOperations, List[Tuple[Enum, torch.Tensor]]] = defaultdict(list)
        for key in self.trackables.get_keys():
            trackable = self.trackables.get_trackable(key)
            reduce_op_to_trackables[trackable.rank_reduce_op].append(trackable)

        for reduce_op, trackables in reduce_op_to_trackables.items():
            if reduce_op == RankReduceOperations.NONE:
                continue
            cloned_values = torch.FloatTensor([trackable.value for trackable in trackables]).cuda()
            reduced_values = Reducer.reduce(tensor=cloned_values, operation=reduce_op).cpu()
            for i, trackable in enumerate(trackables):
                trackable.value = reduced_values[i]
