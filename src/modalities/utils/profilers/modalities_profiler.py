import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel
from torch.profiler import ProfilerActivity, profile, schedule
from tqdm import trange

from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticSteppableComponentIFType
from modalities.main import Main
from modalities.running_env.cuda_env import CudaEnv
from modalities.util import get_experiment_id_from_config, get_synced_experiment_id_of_run
from modalities.utils.logger_utils import get_logger
from modalities.utils.profilers.steppable_components import SteppableComponentIF

logger = get_logger("modalities_profiler")


class InstantiationModel(BaseModel):
    steppable_component: PydanticSteppableComponentIFType


@dataclass
class CustomComponentRegisterable:
    component_key: str
    variant_key: str
    custom_component: type
    custom_config: type


class SteppableProfilerIF:
    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class SteppableMemoryProfiler(SteppableProfilerIF):
    MEMORY_SNAPSHOT_MAX_ENTRIES = 100_000

    def __init__(self, memory_snapshot_path: Path, num_wait_steps: int, num_warmup_steps: int, num_active_steps: int):
        self._memory_snapshot_path = memory_snapshot_path
        self._curr_step = None
        self._num_wait_steps = num_wait_steps
        self._num_warmup_steps = num_warmup_steps
        self._num_active_steps = num_active_steps

    def __enter__(self):
        self._curr_step = 0
        # start recording memory history if there is no wait / warmup steps
        if self._curr_step == self._num_wait_steps + self._num_warmup_steps and self._num_active_steps > 0:
            torch.cuda.memory._record_memory_history(max_entries=SteppableMemoryProfiler.MEMORY_SNAPSHOT_MAX_ENTRIES)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._curr_step is None:
            raise RuntimeError("SteppableMemoryProfilerContext exited without being entered")
        if self._curr_step < self._num_wait_steps + self._num_warmup_steps + self._num_active_steps:
            # if we exit before finishing all steps, dump the memory snapshot
            raise RuntimeError("SteppableMemoryProfilerContext exited before finishing all steps")
        return

    def step(self):
        if self._curr_step is None:
            raise RuntimeError("SteppableMemoryProfilerContext.step() called outside of context manager")
        self._curr_step += 1
        if self._curr_step < self._num_wait_steps + self._num_warmup_steps:
            return
        elif self._curr_step == self._num_wait_steps + self._num_warmup_steps:
            torch.cuda.memory._record_memory_history(max_entries=SteppableMemoryProfiler.MEMORY_SNAPSHOT_MAX_ENTRIES)
        elif (
            self._curr_step == self._num_wait_steps + self._num_warmup_steps + self._num_active_steps
            and self._num_active_steps > 0
        ):
            with open(self._memory_snapshot_path, "wb") as output:
                pickle.dump(torch.cuda.memory._snapshot(), output)


class ProfilerListContext(SteppableProfilerIF):
    def __init__(self, profiler_cms: list[SteppableProfilerIF]):
        self.profiler_cms = profiler_cms
        self._entered = None

    def __enter__(self):
        if self._entered is not None:
            raise RuntimeError("ProfilerListContext entered multiple times without exiting")
        self._entered = []
        for profiler_cm in self.profiler_cms:
            return_val = profiler_cm.__enter__()
            if return_val is not None:
                self._entered.append(return_val)
            else:
                self._entered.append(profiler_cm)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._entered is None:
            raise RuntimeError("ProfilerListContext exited without being entered")
        for profiler_cm in self._entered:
            profiler_cm.__exit__(exc_type, exc_value, traceback)
        self._entered = None

    def step(self):
        if self._entered is None:
            raise RuntimeError("ProfilerListContext.step() called outside of context manager")
        for profiler_cm in self._entered:
            profiler_cm.step()


class ModalitiesProfilerStarter:
    """Starter class to run profiling either in single process or distributed mode."""

    @staticmethod
    def run_distributed(
        config_file_path: Path,
        num_measurement_steps: int,
        num_wait_steps: int,
        num_warmup_steps: int,
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
            num_wait_steps (int): Number of wait steps before profiling starts.
            num_warmup_steps (int): Number of warmup steps before measurement starts.
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
            ModalitiesProfilerStarter._copy_config_to_experiment_folder(
                experiment_root_path=experiment_root_path,
                experiment_id=experiment_id,
                config_file_path=config_file_path,
            )

            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            ModalitiesProfilerStarter._run_helper(
                config_file_path=config_file_path,
                num_measurement_steps=num_measurement_steps,
                num_wait_steps=num_wait_steps,
                num_warmup_steps=num_warmup_steps,
                experiment_folder_path=experiment_root_path / experiment_id,
                global_rank=global_rank,
                world_size=world_size,
                profiled_ranks=profiled_ranks,
                custom_component_registerables=custom_component_registerables,
            )

    @staticmethod
    def run_single_process(
        config_file_path: Path,
        num_measurement_steps: int,
        num_wait_steps: int,
        num_warmup_steps: int,
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
            num_wait_steps (int): Number of wait steps before profiling starts.
            num_warmup_steps (int): Number of warmup steps before measurement starts.
            experiment_root_path (Path): Root path to store experiment results.
            experiment_id (str, optional): Experiment ID. If None, it will be generated.
            custom_component_registerables (list[CustomComponentRegisterable], optional): List of custom
                components to register. Defaults to None.
        """
        if experiment_id is None:
            # get experiment id synched across all ranks
            experiment_id = get_experiment_id_from_config(config_file_path)

        ModalitiesProfilerStarter._copy_config_to_experiment_folder(
            experiment_root_path=experiment_root_path, experiment_id=experiment_id, config_file_path=config_file_path
        )

        global_rank = 0
        world_size = 1
        profiled_ranks = [0]

        ModalitiesProfilerStarter._run_helper(
            config_file_path=config_file_path,
            num_measurement_steps=num_measurement_steps,
            num_wait_steps=num_wait_steps,
            num_warmup_steps=num_warmup_steps,
            experiment_folder_path=experiment_root_path / experiment_id,
            global_rank=global_rank,
            world_size=world_size,
            profiled_ranks=profiled_ranks,
            custom_component_registerables=custom_component_registerables,
        )

    @staticmethod
    def _copy_config_to_experiment_folder(
        experiment_root_path: Path, experiment_id: str, config_file_path: Path
    ) -> None:
        # store a copy of the config file in the experiment folder
        if (
            not torch.distributed.is_initialized()  # copy in single process
            or torch.distributed.is_initialized()  # copy only on rank 0 in distributed process
            and torch.distributed.get_rank() == 0
        ):
            experiment_folder_path = experiment_root_path / experiment_id
            experiment_folder_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_file_path, experiment_folder_path / config_file_path.name)

    @staticmethod
    def _run_helper(
        config_file_path: Path,
        num_measurement_steps: int,
        num_wait_steps: int,
        num_warmup_steps: int,
        experiment_folder_path: Path,
        profiled_ranks: list[int],
        global_rank: int,
        world_size: int,
        custom_component_registerables: list[CustomComponentRegisterable] | None = None,
    ):
        # build profilers
        profiler_activities = [ProfilerActivity.CUDA]  # ProfilerActivity.CPU,
        kernel_profiler = profile(
            activities=profiler_activities,
            schedule=schedule(wait=num_wait_steps, warmup=num_warmup_steps, active=num_measurement_steps),
            record_shapes=False,
            profile_memory=False,
            with_flops=False,
            with_stack=False,
            with_modules=False,
            # record_shapes=True,
            # profile_memory=True,
            # with_flops=True,
            # with_stack=True,
            # with_modules=True,
        )

        SteppableMemoryProfiler(
            memory_snapshot_path=experiment_folder_path / f"memory_snapshot_ranks_{world_size}_rank_{global_rank}.pkl",
            num_wait_steps=num_wait_steps,
            num_warmup_steps=num_warmup_steps,
            num_active_steps=num_measurement_steps,
        )

        profile_context_manager = ProfilerListContext(profiler_cms=[kernel_profiler])  # , memory_profiler]

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
            num_total_steps=num_measurement_steps + num_wait_steps + num_warmup_steps,
            profile_context_manager=profile_context_manager,
            show_progress=(global_rank == profiled_ranks[0]),  # only show progress on a single rank that is profiled
        )
        trace_output_path = experiment_folder_path / f"profiler_trace_ranks_{world_size}_rank_{global_rank}.json"
        summary_output_path = experiment_folder_path / f"profiler_summary_ranks_{world_size}_rank_{global_rank}.txt"

        ModalitiesProfiler.export_profiling_results(
            profiler_context_manager=kernel_profiler,
            trace_output_path=trace_output_path,
            summary_output_path=summary_output_path,
            global_rank=global_rank,
            profiled_ranks=profiled_ranks,
        )


class ModalitiesProfiler:
    @staticmethod
    def profile(
        steppable_component: SteppableComponentIF,
        num_total_steps: int,
        profile_context_manager: SteppableProfilerIF,
        show_progress: bool = False,
    ) -> None:
        """Profile a steppable component using the provided profiler context manager.

        Args:
            steppable_component (SteppableComponentIF): The steppable component to profile.
            num_total_steps (int): Total number of steps to run.
            profile_context_manager (profile): The profiler context manager.
            show_progress (bool): Whether to show a progress bar. Defaults to False.
        """
        if show_progress:
            step_iterator = trange(num_total_steps, desc="Profiling steps")
        else:
            step_iterator = range(num_total_steps)

        with profile_context_manager as profiler:
            for _ in step_iterator:
                steppable_component.step()
                profiler.step()

    @staticmethod
    def export_profiling_results(
        profiler_context_manager: torch.profiler.profile,
        trace_output_path: Path,
        summary_output_path: Path,
        global_rank: int,
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
            logger.info(f"Saving profiling results for rank {global_rank}...")
            profiler_context_manager.export_chrome_trace(trace_output_path.as_posix())
            table = profiler_context_manager.key_averages().table()
            with open(summary_output_path, "w", encoding="utf-8") as f:
                f.write(table)
