import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel
from torch.profiler import ProfilerActivity, profile, schedule

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticSteppableComponentIFType
from modalities.running_env.cuda_env import CudaEnv
from modalities.util import get_experiment_id_from_config, get_synced_experiment_id_of_run
from modalities.utils.profilers.steppable_components import SteppableComponentIF


class InstantiationModel(BaseModel):
    steppable_component: PydanticSteppableComponentIFType


@dataclass
class CustomComponentRegisterable:
    component_key: str
    variant_key: str
    custom_component: type
    custom_config: type


class ModalitiesProfilerStarter:
    """Starter class to run profiling either in single process or distributed mode."""

    @staticmethod
    def run_distributed(
        config_file_path: Path,
        num_measurement_steps: int,
        wait_steps: int,
        warmup_steps: int,
        experiment_root_path: Path,
        profiled_ranks: list[int],
        experiment_id: str | None = None,
        custom_component_registerables: list[CustomComponentRegisterable] | None = None,
    ):
        """Run distributed profiling using the Modalities Profiler.
        This method is primarily intended to run large-scale profiling e.g., for model training.

        Args:
            config_file_path (Path): Path to the configuration file.
            num_measurement_steps (int): Number of measurement steps for profiling.
            wait_steps (int): Number of wait steps before profiling starts.
            warmup_steps (int): Number of warmup steps before measurement starts.
            experiment_root_path (Path): Root path to store experiment results.
            profiled_ranks (list[int]): List of ranks to profile.
            experiment_id (str, optional): Experiment ID. If None, it will be generated. Defaults to None.
            custom_component_registerables (list[CustomComponentRegisterable], optional): List of custom
                components to register. Defaults to None.
        """
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            if experiment_id is None:
                # get experiment id synched across all ranks
                experiment_id = get_synced_experiment_id_of_run(config_file_path)

            # store a copy of the config file in the experiment folder
            if torch.distributed.get_rank() == 0:
                experiment_folder_path = experiment_root_path / experiment_id
                experiment_folder_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(config_file_path, experiment_folder_path / config_file_path.name)

            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            local_rank = int(os.environ["LOCAL_RANK"])

            ModalitiesProfilerStarter._run_helper(
                config_file_path=config_file_path,
                num_measurement_steps=num_measurement_steps,
                wait_steps=wait_steps,
                warmup_steps=warmup_steps,
                experiment_folder_path=experiment_root_path / experiment_id,
                local_rank=local_rank,
                global_rank=global_rank,
                world_size=world_size,
                profiled_ranks=profiled_ranks,
                custom_component_registerables=custom_component_registerables,
            )

    @staticmethod
    def run_single_process(
        config_file_path: Path,
        num_measurement_steps: int,
        wait_steps: int,
        warmup_steps: int,
        experiment_root_path: Path,
        experiment_id: str | None = None,
        custom_component_registerables: list[CustomComponentRegisterable] | None = None,
    ):
        """Run single process profiling using the Modalities Profiler.
        This method is primarily intended for quick profiling experiments
        (e.g., single modules such as custom modules vs Pytorch equivalents) on a single GPU.
        This type of profiling can be seen as a middle ground between distributed profiling and
        single kernel profiling (e.g., using Nsight Compute).

        Args:
            config_file_path (Path): Path to the configuration file.
            num_measurement_steps (int): Number of measurement steps for profiling.
            wait_steps (int): Number of wait steps before profiling starts.
            warmup_steps (int): Number of warmup steps before measurement starts.
            experiment_root_path (Path): Root path to store experiment results.
            experiment_id (str, optional): Experiment ID. If None, it will be generated.
            custom_component_registerables (list[CustomComponentRegisterable], optional): List of custom
                components to register. Defaults to None.
        """
        if experiment_id is None:
            # get experiment id synched across all ranks
            experiment_id = get_experiment_id_from_config(config_file_path)

        # store a copy of the config file in the experiment folder
        experiment_folder_path = experiment_root_path / experiment_id
        experiment_folder_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_file_path, experiment_folder_path / config_file_path.name)

        global_rank = 0
        world_size = 1
        local_rank = 0
        profiled_ranks = [0]

        ModalitiesProfilerStarter._run_helper(
            config_file_path=config_file_path,
            num_measurement_steps=num_measurement_steps,
            wait_steps=wait_steps,
            warmup_steps=warmup_steps,
            experiment_folder_path=experiment_root_path / experiment_id,
            global_rank=global_rank,
            world_size=world_size,
            local_rank=local_rank,
            profiled_ranks=profiled_ranks,
            custom_component_registerables=custom_component_registerables,
        )

    @staticmethod
    def _run_helper(
        config_file_path: Path,
        num_measurement_steps: int,
        wait_steps: int,
        warmup_steps: int,
        experiment_folder_path: Path,
        profiled_ranks: list[int],
        global_rank: int,
        local_rank: int,
        world_size: int,
        custom_component_registerables: list[CustomComponentRegisterable] | None = None,
    ):
        # build profiler
        profiler_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        profile_context_manager = profile(
            activities=profiler_activities,
            schedule=schedule(wait=wait_steps, warmup=warmup_steps, active=num_measurement_steps),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_stack=True,
            with_modules=True,
        )

        # register custom components and build components from config
        # workaround to avoid triggering synchronization of experiment id in single process
        experiment_id = experiment_folder_path.name if world_size == 1 else None
        main_obj = Main(config_file_path, experiment_id=experiment_id)
        if custom_component_registerables is not None:
            for registerable in custom_component_registerables:
                main_obj.add_custom_component(
                    component_key=registerable.component_key,
                    variant_key=registerable.variant_key,
                    custom_component=registerable.custom_component,
                    custom_config=registerable.custom_config,
                )
        components: InstantiationModel = main_obj.build_components(components_model_type=InstantiationModel)

        # run profiling
        ModalitiesProfiler.profile(
            steppable_component=components.steppable_component,
            num_total_steps=num_measurement_steps + wait_steps + warmup_steps,
            profile_context_manager=profile_context_manager,
        )
        trace_output_path = experiment_folder_path / f"profiler_trace_ranks_{world_size}_rank_{global_rank}.json"
        memory_output_path = experiment_folder_path / f"profiler_memory_ranks_{world_size}_rank_{global_rank}.html"
        summary_output_path = experiment_folder_path / f"profiler_summary_ranks_{world_size}_rank_{global_rank}.txt"

        ModalitiesProfiler.export_profiling_results(
            profiler_context_manager=profile_context_manager,
            trace_output_path=trace_output_path,
            memory_output_path=memory_output_path,
            summary_output_path=summary_output_path,
            local_rank=local_rank,
            global_rank=global_rank,
            profiled_ranks=profiled_ranks,
        )


class ModalitiesProfiler:
    @staticmethod
    def profile(
        steppable_component: SteppableComponentIF,
        num_total_steps: int,
        profile_context_manager: profile,
    ) -> None:
        """Profile a steppable component using the provided profiler context manager.

        Args:
            steppable_component (SteppableComponentIF): The steppable component to profile.
            num_total_steps (int): Total number of steps to run.
            profile_context_manager (profile): The profiler context manager.
        """
        with profile_context_manager as profiler:
            for _ in range(num_total_steps):
                steppable_component.step()
                profiler.step()

    @staticmethod
    def export_profiling_results(
        profiler_context_manager: profile,
        trace_output_path: Path,
        memory_output_path: Path,
        summary_output_path: Path,
        global_rank: int,
        local_rank: int,
        profiled_ranks: list[int],
    ) -> None:
        """Export profiling results to specified output paths if the current rank is in profiled_ranks.

        Args:
            profiler_context_manager (profile): The profiler context manager.
            trace_output_path (Path): Path to save the Chrome trace.
            memory_output_path (Path): Path to save the memory timeline.
            summary_output_path (Path): Path to save the summary table.
            global_rank (int): The global rank of the current process.
            local_rank (int): The local rank of the current process.
            profiled_ranks (list[int]): List of ranks to profile.
        """
        if global_rank in profiled_ranks:
            print(f"Saving profiling results for rank {global_rank}...")
            profiler_context_manager.export_chrome_trace(trace_output_path.as_posix())
            device = local_rank if local_rank is not None else None
            profiler_context_manager.export_memory_timeline(memory_output_path.as_posix(), device=device)
            table = profiler_context_manager.key_averages().table()
            with open(summary_output_path, "w", encoding="utf-8") as f:
                f.write(table)
