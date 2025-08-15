# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.

import copy
from typing import Any, Optional, Type

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import PipelineScheduleSingle, get_schedule_class

from modalities.models.parallelism.stages_generator import StagesGenerator
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees


class PipelineFactory:
    """Pipeline factory class to create pipelined models."""

    @staticmethod
    def get_pipelined_model(
        whole_model: nn.Module,
        stages_generator: StagesGenerator,
        device_mesh: DeviceMesh,
        local_rank: int,
        pp_schedule_name: str,
        num_layers_per_stage: int,
    ) -> torch.nn.Module:
        device = torch.device("cuda", local_rank)
        pp_dims = device_mesh[ParallelismDegrees.PP.value].size()

        fqns_per_stage = stages_generator.get_stages(
            num_layers_per_stage=num_layers_per_stage,
            pp_dims=pp_dims,
        )

        pp_mesh = device_mesh[ParallelismDegrees.PP.value]
        schedule_class = get_schedule_class(pp_schedule_name)
        is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
        if not is_single_stage_schedule:
            raise ValueError(
                f"Unsupported pipeline schedule: {pp_schedule_name}. We only support single-stage schedules."
            )
        stage, model = PipelineFactory._get_split_model(
            whole_model=whole_model,
            schedule_class=schedule_class,
            pp_mesh=pp_mesh,
            device=device,
            fqns_per_stage=fqns_per_stage,
        )
        return whole_model  # TODO return pipelined model

    @staticmethod
    def _get_split_model(
        whole_model: nn.Module,
        schedule_class: Type[PipelineScheduleSingle],
        pp_mesh: DeviceMesh,
        device: torch.device,
        fqns_per_stage: list[list[str]],
    ) -> tuple[PipelineStage, nn.Module]:
        def get_stage_id_of_pp_rank(pp_mesh: DeviceMesh):
            # NOTE: torch titan a more complicated way to get the stage id of pp rank
            # since they also allow for multi-stage schedules
            pp_rank = pp_mesh.get_local_rank()
            return pp_rank

        @staticmethod
        def _get_fqn_tree(fqns: list[str]) -> dict[str, Any]:
            fqn_tree = {}
            fqns = set(fqns)  # Ensure unique FQNs
            for fqn in fqns:
                parts = fqn.split(".")
                current_level = fqn_tree
                for part in parts[:-1]:
                    if part not in current_level:
                        current_level[part] = {}
                    elif len(current_level) == 0:
                        raise ValueError(f"Part {part} of {fqn} already exists " "in the tree as a leaf node.")
                    current_level = current_level[part]
                if parts[-1] in current_level:
                    raise ValueError(
                        f" Leaf of {fqn} has already been defined in the tree as an intermediadate node or leaf! "
                        "Cannot replace the existing node as a leaf."
                    )
                current_level[parts[-1]] = {}

            return fqn_tree

        def _build_stage_from_modules(
            fqn_tree: dict[str, Any], module: nn.Module, module_name: Optional[str] = None
        ) -> tuple[PipelineStage, nn.Module]:
            if isinstance(module, nn.ModuleDict):
                if module_name not in fqn_tree:
                    dict_modules = nn.ModuleDict({})
                else:
                    if len(fqn_tree) == 0:
                        # If the module is a leaf node, we can directly use it
                        dict_modules = module
                    else:
                        # If the module is not a leaf node, we need to build a staged module
                        # recursively from the FQN tree
                        dict_modules = {}
                        dict_module_names = [name for name in module.keys() if name in fqn_tree[module_name]]
                        for dict_module_name in dict_module_names:
                            dict_modules[dict_module_name] = _build_stage_from_modules(
                                fqn_tree=fqn_tree[module_name],
                                module=module[dict_module_name],
                                module_name=dict_module_name,
                            )
                        dict_modules = nn.ModuleDict(dict_modules)
                # setattr(module, module_name, dict_modules)
                return dict_modules

            elif isinstance(module, nn.ModuleList):
                if module_name not in fqn_tree:
                    list_modules = nn.ModuleList([])
                else:
                    if len(fqn_tree[module_name]) == 0:
                        # If the module is a leaf node, we can directly use it
                        list_modules = module
                    else:
                        # If the module is not a leaf node, we need to build a staged module
                        # recursively from the FQN tree
                        list_modules = []
                        list_indices = [i for i in range(len(module)) if str(i) in fqn_tree[module_name]]
                        for idx in list_indices:
                            list_modules.append(
                                _build_stage_from_modules(
                                    fqn_tree=fqn_tree[module_name], module=module[idx], module_name=str(idx)
                                )
                            )
                        list_modules = nn.ModuleList(list_modules)
                # setattr(module, module_name, list_modules)
                return list_modules

            else:  # normal nn.Module
                if module_name is not None and module_name not in fqn_tree:
                    # If the module is not in the FQN tree, set it to None
                    return None
                elif module_name is not None and len(fqn_tree[module_name]) == 0:
                    # If the module is a leaf node, we can directly use it
                    return module
                else:
                    # If the module is in the FQN tree, we need to build a staged module
                    # recursively from the FQN tree
                    for module_name, module_value in module.named_children():
                        # If the module is not a leaf node, we need to build a staged module
                        # recursively from the FQN tree
                        staged_module = _build_stage_from_modules(
                            fqn_tree=fqn_tree, module=module_value, module_name=module_name
                        )
                        setattr(module, module_name, staged_module)

                return module

        if not issubclass(schedule_class, PipelineScheduleSingle):
            raise NotImplementedError("Only single-stage schedules are supported for pipeline parallelism.")

        # NOTE: For multi-stage schedule, e.g., Interleaved 1F1B, we have multiple stages per pp rank.
        # This would need to be adapted accordingly in this case.
        stage_idx = get_stage_id_of_pp_rank(pp_mesh)
        module_names = fqns_per_stage[stage_idx]
        whole_model = copy.deepcopy(whole_model)
        fqn_tree = _get_fqn_tree(module_names)
        stage_modules = _build_stage_from_modules(fqn_tree, whole_model)
        stage = PipelineStage(
            submodule=stage_modules,
            stage_index=stage_idx,
            num_stages=len(fqns_per_stage),
            device=device,
            group=pp_mesh.get_group("pp"),
        )
        return stage, whole_model
