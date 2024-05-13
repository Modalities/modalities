from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

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
    tag: Optional[str] = ""


@dataclass
class IntervalState:
    class TrackableCollection:
        def __init__(self):
            # maps TrackableEnum -> Tag -> Trackable
            self.state: Dict[Enum, Dict[str, Trackable]] = defaultdict(dict)

        def set_trackable(self, trackable: Trackable):
            if trackable.key in self.state and trackable.tag in self.state[trackable.key]:
                local_reduce_op_fun = trackable.local_reduce_op.value
                old_trackable = self.state[trackable.key][trackable.tag]
                self.state[trackable.key][trackable.tag].value = local_reduce_op_fun(
                    [old_trackable.value, trackable.value]
                )
            else:
                self.state[trackable.key][trackable.tag] = trackable

        def get_trackable(self, key: Enum, tag: str) -> Trackable:
            return self.state[key][tag]

        def get_tags(self, key: Enum) -> List[str]:
            return list(self.state[key].keys())

        def get_keys(self) -> List[Enum]:
            return list(self.state.keys())

    @dataclass
    class MetaInformation:
        step_id: int
        num_steps: int
        dataloader_tag: str
        experiment_status: ExperimentStatus

    meta_information: MetaInformation
    trackables: TrackableCollection = TrackableCollection()

    def reduce_values_across_ranks(self) -> Dict[Enum, torch.Tensor]:
        reduce_op_to_trackables: Dict[RankReduceOperations, List[Trackable]] = defaultdict(list)
        # get the trackables and group them by the rank reduce operation
        for key in self.trackables.get_keys():
            tags = self.trackables.get_tags(key=key)
            for tag in tags:
                trackable = self.trackables.get_trackable(key=key, tag=tag)
                reduce_op_to_trackables[trackable.rank_reduce_op].append(trackable)

        # reduce the trackables for each reduce operation
        for reduce_op, trackables in reduce_op_to_trackables.items():
            if reduce_op == RankReduceOperations.NONE:
                continue
            cloned_values = torch.FloatTensor([trackable.value for trackable in trackables]).cuda()
            reduced_values = Reducer.reduce(tensor=cloned_values, operation=reduce_op).cpu()
            for i, trackable in enumerate(trackables):
                trackable.value = reduced_values[i]
