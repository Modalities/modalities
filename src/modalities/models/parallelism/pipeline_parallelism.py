# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.

import copy
import os
from enum import Enum
from typing import Any, Optional, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import PipelineScheduleSingle, get_schedule_class

from modalities.loss_functions import Loss
from modalities.models.parallelism.stages_generator import StagesGenerator
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from modalities.utils.logger_utils import get_logger

logger = get_logger(__name__)


class Pipeline:
    def __init__(
        self,
        pp_stage: PipelineStage,
        model_part: nn.Module,
        pp_schedule: Optional[PipelineScheduleSingle] = None,
    ):
        self._pp_stage = pp_stage
        self._model_part = model_part
        self._pp_schedule = pp_schedule

    @property
    def is_first_pp_stage(self) -> bool:
        return self._pp_stage.is_first

    @property
    def is_last_pp_stage(self) -> bool:
        return self._pp_stage.is_last

    @property
    def pp_stage(self) -> PipelineStage:
        return self._pp_stage

    @property
    def model_part(self) -> nn.Module:
        return self._model_part

    @property
    def pp_schedule(self) -> Optional[PipelineScheduleSingle]:
        return self._pp_schedule

    @pp_schedule.setter
    def pp_schedule(self, schedule: PipelineScheduleSingle):
        self._pp_schedule = schedule


class PipelineSelectionTypes(Enum):
    """Enum for pipeline selection types."""

    PP_STAGE = "PP_STAGE"
    MODEL_PART = "MODEL_PART"
    PP_SCHEDULE = "PP_SCHEDULE"


class ComponentSelectorFromPipeline:
    @staticmethod
    def select(pipeline: Pipeline, selection_type: PipelineSelectionTypes) -> Any:
        """Selects a component from the pipeline based on the selection type."""
        if selection_type == PipelineSelectionTypes.PP_STAGE:
            return pipeline.pp_stage
        elif selection_type == PipelineSelectionTypes.MODEL_PART:
            return pipeline.model_part
        elif selection_type == PipelineSelectionTypes.PP_SCHEDULE:
            return pipeline.pp_schedule
        else:
            raise ValueError(f"Unsupported selection type: {selection_type}")


# Global guard to avoid rebuilding the pipeline multiple times per process
_PIPELINE_BUILT = False


def _mark_pipeline_built():
    global _PIPELINE_BUILT
    if _PIPELINE_BUILT and os.environ.get("MODALITIES_DEBUG_PIPELINE", "0") == "1":
        print(f"[PP-DEBUG] Pipeline already built on rank {dist.get_rank()}", flush=True)
    _PIPELINE_BUILT = True


class PipelineFactory:
    """Pipeline factory class to create pipelined models."""

    @staticmethod
    def get_pipeline(
        pp_stage: PipelineStage, model_part: nn.Module, pp_schedule: Optional[PipelineScheduleSingle] = None
    ) -> Pipeline:
        return Pipeline(pp_stage=pp_stage, model_part=model_part, pp_schedule=pp_schedule)

    @staticmethod
    def get_staged_pipeline(
        whole_model: nn.Module,
        stages_generator: StagesGenerator,
        device_mesh: DeviceMesh,
        local_rank: int,
        pp_schedule_name: str,
        num_layers_per_stage: int,
        copy_model: bool = True,  # allow disabling deep copy for debugging
    ) -> Pipeline:
        device = torch.device("cuda", local_rank)
        pp_dims = device_mesh[ParallelismDegrees.PP.value].size()

        fqns_per_stage = stages_generator.get_stages(
            num_layers_per_stage=num_layers_per_stage,
            pp_dims=pp_dims,
        )

        schedule_class = get_schedule_class(pp_schedule_name)
        if not issubclass(schedule_class, PipelineScheduleSingle):
            raise ValueError(
                f"Unsupported pipeline schedule: {pp_schedule_name}. Only single-stage schedules are supported."
            )

        pp_stage, model_part = PipelineFactory._get_split_model(
            whole_model=whole_model,
            schedule_class=schedule_class,
            device_mesh=device_mesh,  # pass full mesh
            device=device,
            fqns_per_stage=fqns_per_stage,
            copy_model=copy_model,
        )

        _mark_pipeline_built()
        if os.environ.get("MODALITIES_DEBUG_PIPELINE", "0") == "1":
            coord = device_mesh.get_coordinate()
            fqns_this_stage = fqns_per_stage[pp_stage.stage_index]
            debug_msg = (
                f"[PP-DEBUG] rank={dist.get_rank()} coord={coord} "
                f"stage_idx={pp_stage.stage_index} "
                f"fqns={fqns_this_stage}"
            )
            print(debug_msg, flush=True)

        pipeline = Pipeline(pp_stage=pp_stage, model_part=model_part)
        dist.barrier()
        return pipeline

    @staticmethod
    def _get_split_model(
        whole_model: nn.Module,
        schedule_class: Type[PipelineScheduleSingle],
        device_mesh: DeviceMesh,
        device: torch.device,
        fqns_per_stage: list[list[str]],
        copy_model: bool,
    ) -> tuple[PipelineStage, nn.Module]:
        """
        Build the single PipelineStage and its model partition for the current rank.
        """
        # Determine stage index from global device mesh coordinate
        if ParallelismDegrees.PP.value not in device_mesh.mesh_dim_names:
            raise RuntimeError("Pipeline dimension 'pp' not found in device mesh.")
        pp_axis = list(device_mesh.mesh_dim_names).index(ParallelismDegrees.PP.value)
        stage_idx = device_mesh.get_coordinate()[pp_axis]

        # Optionally deepcopy the whole model (disable for debugging shared weights)
        model_root = copy.deepcopy(whole_model) if copy_model else whole_model

        # ---- Build FQN tree utilities (unchanged logic, just moved) ----
        def _get_fqn_tree(fqns: list[str]) -> dict[str, Any]:
            fqn_tree: dict[str, Any] = {}
            for fqn in set(fqns):
                parts = fqn.split(".")
                cur = fqn_tree
                for part in parts[:-1]:
                    cur = cur.setdefault(part, {})
                if parts[-1] in cur:
                    raise ValueError(f"Duplicate leaf {fqn}")
                cur[parts[-1]] = {}
            return fqn_tree

        def _build_stage_from_modules(
            fqn_tree: dict[str, Any], module: nn.Module, module_name: Optional[str] = None
        ) -> nn.Module | None:
            # Prune modules not in this stage; keep structure minimal
            if isinstance(module, nn.ModuleDict):
                if module_name and module_name not in fqn_tree:
                    return None
                keys = list(module.keys()) if module_name is None else fqn_tree[module_name].keys()
                new_items = {}
                for k in keys:
                    subtree = fqn_tree[k] if module_name is None else fqn_tree[module_name][k]
                    built = _build_stage_from_modules(subtree, module[k], k)
                    if built is not None:
                        new_items[k] = built
                return nn.ModuleDict(new_items)
            elif isinstance(module, nn.ModuleList):
                if module_name and module_name not in fqn_tree:
                    return None
                indices = range(len(module)) if module_name is None else [int(i) for i in fqn_tree[module_name].keys()]
                new_list = []
                for i in indices:
                    key = str(i)
                    subtree = fqn_tree[key] if module_name is None else fqn_tree[module_name][key]
                    built = _build_stage_from_modules(subtree, module[i], key)
                    if built is not None:
                        new_list.append(built)
                return nn.ModuleList(new_list)
            else:
                # Leaf or inner module
                if module_name is not None:
                    if module_name not in fqn_tree:
                        return None
                    if len(fqn_tree[module_name]) == 0:
                        return module
                    # Recurse into children
                    for child_name, child in list(module.named_children()):
                        built = _build_stage_from_modules(fqn_tree[module_name], child, child_name)
                        if built is None:
                            delattr(module, child_name)
                        else:
                            setattr(module, child_name, built)
                    return module
                else:
                    # Root call
                    for child_name, child in list(module.named_children()):
                        # If child_name is top-level in tree
                        if child_name in fqn_tree:
                            subtree = fqn_tree[child_name]
                            built = _build_stage_from_modules({child_name: subtree}, child, child_name)
                            if built is None:
                                delattr(module, child_name)
                            else:
                                setattr(module, child_name, built)
                        else:
                            # Remove modules not in this stage
                            delattr(module, child_name)
                    return module

        module_names = fqns_per_stage[stage_idx]
        fqn_tree = _get_fqn_tree(module_names)
        stage_modules = _build_stage_from_modules(fqn_tree, model_root)

        # Obtain proper PP communication group from full mesh
        pp_group = device_mesh.get_group(ParallelismDegrees.PP.value)

        stage = PipelineStage(
            submodule=stage_modules,
            stage_index=stage_idx,
            num_stages=len(fqns_per_stage),
            device=device,
            group=pp_group,
        )
        return stage, stage_modules

    @staticmethod
    def get_scheduled_pipeline(
        loss_fn: Loss, pp_schedule_name: str, batch_size: int, microbatch_size: int, pp_degree: int, pipeline: Pipeline
    ) -> Pipeline:
        # TODO: Addd validation in config that batch_size is divisible by microbatch_size
        # and n_microbatches must be >= pp_degree
        n_microbatches = batch_size // microbatch_size
        num_total_stages = pp_degree
        pp_schedule_class = get_schedule_class(pp_schedule_name)
        pp_schedule = pp_schedule_class(
            stage=pipeline.pp_stage,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
        )
        logger.info(
            f"Using pipeline schedule {pp_schedule} with {n_microbatches} microbatches and {num_total_stages} stages."
        )
        pipeline.pp_schedule = pp_schedule
        return pipeline
