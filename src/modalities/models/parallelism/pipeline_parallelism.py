# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.

import copy
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Type, cast

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)

from modalities.loss_functions import Loss
from modalities.models.model import NNModel
from modalities.models.parallelism.stages_generator import StagesGenerator
from modalities.running_env.env_utils import PyTorchDtypes
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from modalities.utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StaticStageIOSpec:
    """Shapes and dtype of the tensors crossing pipeline stage boundaries, used to build STATIC
    pipeline metadata (see PipelineFactory.get_staged_pipeline).

    ``activation_dtype`` must be the dtype of the model's actual activations (i.e. the
    mixed-precision ``param_dtype``); the pipeline allocates its P2P buffers from it, so a mismatch
    surfaces as a shape/dtype validation error or a corrupted inter-stage transfer.
    """

    microbatch_size: int
    sequence_length: int
    hidden_dim: int
    vocab_size: int
    activation_dtype: torch.dtype


class Pipeline:
    def __init__(
        self,
        pp_stages: Iterable[PipelineStage],
        model_parts: Iterable[nn.Module],
        pp_schedule: Optional[PipelineScheduleSingle | PipelineScheduleMulti] = None,
    ):
        self._pp_stages = list(pp_stages)
        self._model_parts = list(model_parts)
        self._pp_schedule = pp_schedule

    @property
    def has_first_pp_stage(self) -> bool:
        return any(stage.is_first for stage in self._pp_stages)

    @property
    def has_last_pp_stage(self) -> bool:
        return any(stage.is_last for stage in self._pp_stages)

    @property
    def pp_stages(self) -> list[PipelineStage]:
        return self._pp_stages

    @property
    def model_parts(self) -> list[nn.Module]:
        return self._model_parts

    @property
    def pp_schedule(self) -> Optional[PipelineScheduleSingle | PipelineScheduleMulti]:
        return self._pp_schedule

    @pp_schedule.setter
    def pp_schedule(self, schedule: PipelineScheduleSingle | PipelineScheduleMulti):
        self._pp_schedule = schedule


class PipelineSelectionTypes(Enum):
    """Enum for pipeline selection types."""

    PP_STAGE = "PP_STAGE"
    MODEL_PART = "MODEL_PART"
    PP_SCHEDULE = "PP_SCHEDULE"


class ComponentSelectorFromPipeline:
    @staticmethod
    def select(
        pipeline: Pipeline, selection_type: PipelineSelectionTypes
    ) -> list[PipelineStage] | list[nn.Module] | PipelineScheduleSingle | PipelineScheduleMulti | None:
        """Selects a component from the pipeline based on the selection type."""
        if selection_type == PipelineSelectionTypes.PP_STAGE:
            return pipeline.pp_stages
        elif selection_type == PipelineSelectionTypes.MODEL_PART:
            return pipeline.model_parts
        elif selection_type == PipelineSelectionTypes.PP_SCHEDULE:
            return pipeline.pp_schedule
        else:
            raise ValueError(f"Unsupported selection type: {selection_type}")


class PipelineFactory:
    """Pipeline factory class to create pipelined models."""

    @staticmethod
    def get_pipeline(
        pp_stages: list[PipelineStage], model_parts: list[NNModel], pp_schedule: Optional[PipelineScheduleSingle] = None
    ) -> Pipeline:
        return Pipeline(pp_stages=pp_stages, model_parts=model_parts, pp_schedule=pp_schedule)

    @staticmethod
    def get_staged_pipeline(
        whole_model: NNModel,
        stages_generator: StagesGenerator,
        device_mesh: DeviceMesh,
        local_rank: int,
        pp_schedule_name: str,
        num_layers_per_stage: int,
        static_io_microbatch_size: Optional[int] = None,
        static_io_sequence_length: Optional[int] = None,
        static_io_hidden_dim: Optional[int] = None,
        static_io_vocab_size: Optional[int] = None,
        static_io_dtype: Optional[PyTorchDtypes] = None,
    ) -> Pipeline:
        """Builds the pipeline stages for the local PP rank.

        Static pipeline metadata (opt-in): if all ``static_io_*`` arguments are provided, each
        PipelineStage is constructed with explicit example ``input_args``/``output_args`` so that
        PyTorch runs the pipeline in STATIC metadata mode and SKIPS its live shape-inference pass
        (a dummy forward+backward through every stage). That live pass is unsafe when a stage
        contains its own internal collective -- e.g. expert-parallel (EP) all-to-all in the MoE
        FFN -- because the dummy forward/backward and the pipeline's P2P metadata exchange interleave
        across the EP and PP communicators and deadlock. Providing static metadata avoids running the
        dummy pass at all, so real training starts with all ranks in lockstep. This is what makes EP
        composable with PP; without it, EP+PP hangs during pipeline initialization. When the arguments
        are omitted (e.g. the GPT2 path), the pipeline falls back to the default DYNAMIC inference.
        ``static_io_dtype`` must be the dtype of the actual inter-stage activations (i.e. the
        mixed-precision ``param_dtype`` of the model), because it determines the dtype of the P2P
        buffers the pipeline allocates. StagedPipelineConfig enforces the all-or-none rule and
        rejects EP+PP configurations without static metadata.

        Note: the inter-stage activations must be plain tensors (not DTensors) for STATIC mode to be
        sufficient without also supplying gradient metadata. This holds here because tensor
        parallelism keeps the residual stream as local tensors (use_local_output=True).
        """
        device = torch.device("cuda", local_rank)
        pp_dims = device_mesh[ParallelismDegrees.PP.value].size()

        fqns_per_stage = stages_generator.get_stages(
            num_layers_per_stage=num_layers_per_stage,
            pp_dims=pp_dims,
        )

        pp_mesh = device_mesh[ParallelismDegrees.PP.value]
        schedule_class: Type[PipelineScheduleSingle | PipelineScheduleMulti] = get_schedule_class(pp_schedule_name)

        static_io_settings = (
            static_io_microbatch_size,
            static_io_sequence_length,
            static_io_hidden_dim,
            static_io_vocab_size,
            static_io_dtype,
        )
        if all(setting is not None for setting in static_io_settings):
            static_io_spec = StaticStageIOSpec(
                microbatch_size=static_io_microbatch_size,
                sequence_length=static_io_sequence_length,
                hidden_dim=static_io_hidden_dim,
                vocab_size=static_io_vocab_size,
                activation_dtype=static_io_dtype.value,
            )
        elif any(setting is not None for setting in static_io_settings):
            # Defense in depth: StagedPipelineConfig already rejects partial specifications, but this
            # factory is also callable directly and a silent fallback to dynamic inference would
            # deadlock under EP (see docstring).
            raise ValueError(
                "Static pipeline I/O metadata must be specified either completely or not at all, "
                "but only some of the static_io_* arguments were provided."
            )
        else:
            static_io_spec = None

        pp_stages, model_parts = PipelineFactory._get_split_model(
            whole_model=whole_model,
            schedule_class=schedule_class,
            pp_mesh=pp_mesh,
            device=device,
            fqns_per_stage=fqns_per_stage,
            static_io_spec=static_io_spec,
        )

        pipeline = Pipeline(pp_stages=pp_stages, model_parts=model_parts)
        return pipeline

    @staticmethod
    def _build_static_stage_io(
        stage_idx: int,
        num_stages: int,
        static_io_spec: StaticStageIOSpec,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds example (input_args, output_args) tensors describing a stage's I/O for STATIC
        pipeline metadata. The first stage consumes token ids (int64), every inter-stage boundary
        carries the hidden state, and the last stage emits logits. The activation dtype comes from
        the caller (``static_io_dtype``) and must match the model's actual activation dtype, since
        the pipeline sizes its P2P buffers from it.

        The float activation examples set requires_grad=True: in STATIC mode PyTorch derives the
        backward gradient metadata (and thus whether to allocate the cross-stage gradient P2P
        buffers) from the requires_grad flag of these forward examples. Without it, no gradient is
        sent back and a non-first stage's backward receives a None output-gradient. Token ids are
        integer and therefore always requires_grad=False."""
        microbatch_size = static_io_spec.microbatch_size
        sequence_length = static_io_spec.sequence_length
        hidden_dim = static_io_spec.hidden_dim
        vocab_size = static_io_spec.vocab_size
        activation_dtype = static_io_spec.activation_dtype

        is_first = stage_idx == 0
        is_last = stage_idx == num_stages - 1

        if is_first:
            input_args = torch.zeros((microbatch_size, sequence_length), dtype=torch.long, device=device)
        else:
            input_args = torch.zeros(
                (microbatch_size, sequence_length, hidden_dim), dtype=activation_dtype, device=device
            ).requires_grad_(True)

        if is_last:
            output_args = torch.zeros(
                (microbatch_size, sequence_length, vocab_size), dtype=activation_dtype, device=device
            ).requires_grad_(True)
        else:
            output_args = torch.zeros(
                (microbatch_size, sequence_length, hidden_dim), dtype=activation_dtype, device=device
            ).requires_grad_(True)

        return input_args, output_args

    @staticmethod
    def _get_split_model(
        whole_model: NNModel,
        schedule_class: Type[PipelineScheduleSingle | PipelineScheduleMulti],
        pp_mesh: DeviceMesh,
        device: torch.device,
        fqns_per_stage: list[list[str]],
        static_io_spec: Optional[StaticStageIOSpec] = None,
    ) -> tuple[list[PipelineStage], list[NNModel]]:
        num_stages = len(fqns_per_stage)
        stage_indices = PipelineFactory._get_stage_ids_of_pp_rank(pp_mesh, num_stages, schedule_class)
        stages, stage_modules = zip(
            *(
                PipelineFactory._build_model_part_for_stage(
                    whole_model, pp_mesh, device, fqns_per_stage, stage_idx, static_io_spec
                )
                for stage_idx in stage_indices
            )
        )
        return list(stages), list(stage_modules)

    @staticmethod
    def _get_stage_ids_of_pp_rank(
        pp_mesh: DeviceMesh,
        num_stages: int,
        schedule_class: Type[PipelineScheduleSingle | PipelineScheduleMulti],
    ) -> list[int]:
        style = "v" if schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV) else "loop"
        pp_size = pp_mesh.size()
        pp_rank = pp_mesh.get_local_rank()
        stages_per_rank = num_stages // pp_size
        if style == "loop":
            return [pp_rank + s * pp_size for s in range(stages_per_rank)]
        elif style == "v":
            if stages_per_rank != 2:
                raise ValueError(f"v schedules assume 2 stages per rank but got {stages_per_rank}.")
            stage_v_pairs = list(zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1)))
            return list(stage_v_pairs[pp_rank])
        else:
            raise ValueError(f"Unsupported schedule style: {style}")

    @staticmethod
    def _build_model_part_for_stage(
        whole_model: NNModel,
        pp_mesh: DeviceMesh,
        device: torch.device,
        fqns_per_stage: list[list[str]],
        stage_idx: int,
        static_io_spec: Optional[StaticStageIOSpec] = None,
    ) -> tuple[PipelineStage, NNModel]:
        num_stages = len(fqns_per_stage)
        module_names = fqns_per_stage[stage_idx]
        whole_model = copy.deepcopy(whole_model)
        fqn_tree = PipelineFactory._get_fqn_tree(module_names)
        stage_modules = PipelineFactory._build_stage_from_modules(fqn_tree, whole_model)
        stage_modules = cast(NNModel, stage_modules)
        PipelineFactory._filter_weight_decay_groups_(stage_modules)

        # When a static I/O spec is provided, supply explicit example input/output tensors so the
        # pipeline uses STATIC metadata mode and skips its live shape-inference pass (see
        # get_staged_pipeline for why this is required for EP+PP). Otherwise fall back to DYNAMIC.
        input_args, output_args = (
            PipelineFactory._build_static_stage_io(stage_idx, num_stages, static_io_spec, device)
            if static_io_spec is not None
            else (None, None)
        )

        stage = PipelineStage(
            submodule=stage_modules,
            stage_index=stage_idx,
            num_stages=num_stages,
            device=device,
            input_args=input_args,
            output_args=output_args,
            group=pp_mesh.get_group("pp"),
        )

        return stage, stage_modules

    @staticmethod
    def _get_fqn_tree(fqns: list[str]) -> dict[str, Any]:
        fqn_tree: dict[str, Any] = {}
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
                    f" Leaf of {fqn} has already been defined in the tree as an intermediate node or leaf! "
                    "Cannot replace the existing node as a leaf."
                )
            current_level[parts[-1]] = {}

        return fqn_tree

    @staticmethod
    def _build_stage_from_modules(
        fqn_tree: dict[str, Any], module: nn.Module, module_name: Optional[str] = None
    ) -> nn.Module:
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
                        dict_modules[dict_module_name] = PipelineFactory._build_stage_from_modules(
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
                            PipelineFactory._build_stage_from_modules(
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
                    staged_module = PipelineFactory._build_stage_from_modules(
                        fqn_tree=fqn_tree, module=module_value, module_name=module_name
                    )
                    setattr(module, module_name, staged_module)

            return module

    @staticmethod
    def _filter_weight_decay_groups_(stage_modules: NNModel):
        params = {name for name, parameter in stage_modules.named_parameters() if parameter.requires_grad}
        for group_list in stage_modules.weight_decay_groups.values():
            remove_from_group = [
                group_entry
                for group_entry in group_list
                if all([not bool(re.search(group_entry, name)) for name in params])
            ]
            for remove in remove_from_group:
                group_list.remove(remove)
        empty_group_keys = [k for k, v in stage_modules.weight_decay_groups.items() if len(v) == 0]
        for key in empty_group_keys:
            del stage_modules.weight_decay_groups[key]

    @staticmethod
    def get_scheduled_pipeline(
        loss_fn: Loss, pp_schedule_name: str, batch_size: int, microbatch_size: int, pp_degree: int, pipeline: Pipeline
    ) -> Pipeline:
        # TODO: Addd validation in config that batch_size is divisible by microbatch_size
        # and n_microbatches must be >= pp_degree
        n_microbatches = batch_size // microbatch_size
        num_total_stages = pp_degree * len(pipeline.pp_stages)
        pp_schedule = PipelineFactory._build_pp_schedule(loss_fn, pp_schedule_name, n_microbatches, pipeline.pp_stages)
        logger.info(
            f"Using pipeline schedule {pp_schedule} with {n_microbatches} microbatches and {num_total_stages} stages."
        )
        pipeline.pp_schedule = pp_schedule
        return pipeline

    @staticmethod
    def _build_pp_schedule(
        loss_fn: Loss,
        pp_schedule_name: str,
        n_microbatches: int,
        pp_stage_or_stages: PipelineStage | list[PipelineStage],
    ) -> PipelineScheduleSingle | PipelineScheduleMulti:
        pp_schedule_class: Type[PipelineScheduleSingle | PipelineScheduleMulti] = get_schedule_class(pp_schedule_name)
        if issubclass(pp_schedule_class, PipelineScheduleSingle):
            if isinstance(pp_stage_or_stages, list):
                assert len(pp_stage_or_stages) == 1, (
                    f"Expected a single PipelineStage for single-stage schedule "
                    f"but got {len(pp_stage_or_stages)} stages."
                )
                pp_stage_or_stages = pp_stage_or_stages[0]
            pp_schedule = pp_schedule_class(
                stage=pp_stage_or_stages,
                n_microbatches=n_microbatches,
                loss_fn=loss_fn,
            )
        elif issubclass(pp_schedule_class, PipelineScheduleMulti):
            assert isinstance(pp_stage_or_stages, list), "Expected a list of PipelineStages for multi-stage schedule."
            pp_schedule = pp_schedule_class(
                stages=pp_stage_or_stages,
                n_microbatches=n_microbatches,
                loss_fn=loss_fn,
            )
        else:
            raise ValueError(f"Unsupported pipeline schedule class: {pp_schedule_class}.")
        return pp_schedule
