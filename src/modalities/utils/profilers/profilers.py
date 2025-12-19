import pickle
from pathlib import Path

import torch
from torch.profiler import profile, schedule

from modalities.utils.logger_utils import get_logger

logger = get_logger("modalities_profiler")


class SteppableProfilerIF:
    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SteppableCombinedProfiler(SteppableProfilerIF):
    def __init__(self, profilers: list[SteppableProfilerIF]):
        self._profilers = profilers
        self._entered = None

    def __enter__(self):
        if self._entered is not None:
            raise RuntimeError("ProfilerListContext entered multiple times without exiting")
        self._entered = []
        for profiler_cm in self._profilers:
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

    def __len__(self):
        max_len = max([len(p) for p in self._profilers])
        min_len = min([len(p) for p in self._profilers if not isinstance(p, SteppableNoProfiler)])
        if max_len != min_len:
            logger.warning(
                "SteppableCombinedProfiler has profilers of different step lengths."
                f" Max steps: {max_len}, Min steps: {min_len}."
                " The combined profiler will run for the maximum steps, and some profilers may be inactive or fail."
            )
        return max_len

    def step(self):
        if self._entered is None:
            raise RuntimeError("ProfilerListContext.step() called outside of context manager")
        for profiler_cm in self._entered:
            profiler_cm.step()


class SteppableNoProfiler(SteppableProfilerIF):
    def __init__(self, num_steps: int) -> None:
        super().__init__()
        self._num_steps = num_steps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    def __len__(self):
        return self._num_steps

    def step(self):
        return


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

    def __len__(self):
        return self._num_wait_steps + self._num_warmup_steps + self._num_active_steps

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


class SteppableKernelProfiler(SteppableProfilerIF):
    def __init__(
        self,
        num_wait_steps: int,
        num_warmup_steps: int,
        num_active_steps: int,
        profiler_activities: list[torch.profiler.ProfilerActivity],
        record_shapes: bool,
        profile_memory: bool,
        with_flops: bool,
        with_stack: bool,
        with_modules: bool,
        trace_output_path: Path,
        summary_output_path: Path,
    ) -> None:  # TODO specify Callable type
        super().__init__()
        self._num_wait_steps = num_wait_steps
        self._num_warmup_steps = num_warmup_steps
        self._num_active_steps = num_active_steps
        self._profiler_activities = profiler_activities
        self._record_shapes = record_shapes
        self._profile_memory = profile_memory
        self._with_flops = with_flops
        self._with_stack = with_stack
        self._with_modules = with_modules
        self._trace_output_path = trace_output_path
        self._summary_output_path = summary_output_path
        self._kernel_profiler = None

    def __enter__(self):
        if self._kernel_profiler is not None:
            raise RuntimeError("Context entered multiple times without exiting")
        self._curr_step = 0
        self._kernel_profiler = profile(
            activities=self._profiler_activities,
            schedule=schedule(wait=self._num_wait_steps, warmup=self._num_warmup_steps, active=self._num_active_steps),
            record_shapes=self._record_shapes,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
            with_stack=self._with_stack,
            with_modules=self._with_modules,
        )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._export_profiling_results()
        self._kernel_profiler = None
        if self._curr_step is None:
            raise RuntimeError("SteppableKernelProfiler exited without being entered")
        if self._curr_step < self._num_wait_steps + self._num_warmup_steps + self._num_active_steps:
            # if we exit before finishing all steps, dump the memory snapshot
            raise RuntimeError("SteppableKernelProfiler exited before finishing all steps")
        return

    def __len__(self):
        return self._num_wait_steps + self._num_warmup_steps + self._num_active_steps

    def step(self):
        if self._curr_step is None:
            raise RuntimeError("SteppableKernelProfiler.step() called outside of context manager")
        if self._kernel_profiler is None:
            raise RuntimeError("SteppableKernelProfiler.step() called when profiler is not initialized")
        self._curr_step += 1
        self._kernel_profiler.step()

    def _export_profiling_results(self) -> None:
        # Export profiling results to specified output paths if the current rank is in profiled_ranks.
        if self._kernel_profiler is None:
            raise RuntimeError(
                "SteppableKernelProfiler._export_profiling_results() called when profiler is not initialized"
            )
        logger.info("Saving profiling results...")
        self._kernel_profiler.export_chrome_trace(self._trace_output_path.as_posix())
        table = self._kernel_profiler.key_averages().table()
        self._summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._summary_output_path, "w", encoding="utf-8") as f:
            f.write(table)
