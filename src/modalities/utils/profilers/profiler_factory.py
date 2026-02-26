from pathlib import Path

import torch

from modalities.utils.profilers.profiler_configs import ModalitiesProfilerActivity
from modalities.utils.profilers.profilers import (
    SteppableKernelProfiler,
    SteppableMemoryProfiler,
    SteppableNoProfiler,
    SteppableProfilerIF,
)


class ProfilerFactory:
    """Factory class to create different types of profilers based on the provided settings."""

    @staticmethod
    def create_steppable_kernel_profiler(
        num_wait_steps: int,
        num_warmup_steps: int,
        num_active_steps: int,
        profiler_activities: list[ModalitiesProfilerActivity],
        profile_memory: bool,
        record_shapes: bool,
        with_flops: bool,
        with_stack: bool,
        with_modules: bool,
        output_folder_path: Path,
        tracked_ranks: list[int] | None = None,
    ) -> SteppableProfilerIF:
        """Creates a steppable kernel profiler based on the provided settings."""
        if tracked_ranks is None:
            tracked_ranks = []
        global_rank, world_size = ProfilerFactory._get_global_rank_and_world_size()

        profiler_activities_converted = []
        for activity in profiler_activities:
            if activity == ModalitiesProfilerActivity.CPU:
                profiler_activities_converted.append(torch.profiler.ProfilerActivity.CPU)
            elif activity == ModalitiesProfilerActivity.CUDA:
                profiler_activities_converted.append(torch.profiler.ProfilerActivity.CUDA)

        trace_output_path = output_folder_path / f"profiler_trace_ranks_{world_size}_rank_{global_rank}.json"
        summary_output_path = output_folder_path / f"profiler_summary_ranks_{world_size}_rank_{global_rank}.txt"

        profiler = SteppableKernelProfiler(
            num_wait_steps=num_wait_steps,
            num_warmup_steps=num_warmup_steps,
            num_active_steps=num_active_steps,
            profiler_activities=profiler_activities_converted,
            profile_memory=profile_memory,
            record_shapes=record_shapes,
            with_flops=with_flops,
            with_stack=with_stack,
            with_modules=with_modules,
            trace_output_path=trace_output_path,
            summary_output_path=summary_output_path,
        )

        if global_rank not in tracked_ranks:
            num_steps = len(profiler)
            return SteppableNoProfiler(num_steps=num_steps)
        else:
            return profiler

    @staticmethod
    def create_steppable_memory_profiler(
        memory_snapshot_folder_path: Path,
        num_wait_steps: int,
        num_warmup_steps: int,
        num_active_steps: int,
        tracked_ranks: list[int] | None = None,
    ) -> SteppableProfilerIF:
        """Creates a steppable memory profiler based on the provided settings."""
        if tracked_ranks is None:
            tracked_ranks = []

        global_rank, world_size = ProfilerFactory._get_global_rank_and_world_size()
        profiler = SteppableMemoryProfiler(
            memory_snapshot_path=memory_snapshot_folder_path
            / f"memory_snapshot_ranks_{world_size}_rank_{global_rank}.pkl",
            num_wait_steps=num_wait_steps,
            num_warmup_steps=num_warmup_steps,
            num_active_steps=num_active_steps,
        )
        if global_rank not in tracked_ranks:
            num_steps = len(profiler)
            return SteppableNoProfiler(num_steps=num_steps)
        else:
            return profiler

    @staticmethod
    def _get_global_rank_and_world_size() -> tuple[int, int]:
        global_rank = 0
        world_size = 1
        if torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        return global_rank, world_size
