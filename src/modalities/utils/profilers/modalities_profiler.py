import hashlib
import json
import os
import socket
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from pydantic import BaseModel

from modalities.batch import DatasetBatch, InferenceResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydanctic_if_types import (
    PydanticDatasetBatchGeneratorIFType,
    PydanticFSDP2ModuleType,
    PydanticLossIFType,
    PydanticOptimizerIFType,
)
from modalities.loss_functions import Loss
from modalities.main import Main
from modalities.running_env.cuda_env import CudaEnv
from modalities.util import get_synced_string
from modalities.utils.profilers.batch_generator import DatasetBatchGeneratorIF
from modalities.utils.profilers.grid_search_utils import ConfigValue
from modalities.utils.typing_utils import FSDPX


class InstantiationModel(BaseModel):
    initialized_model: PydanticFSDP2ModuleType
    loss_fn: PydanticLossIFType
    optimizer: Optional[PydanticOptimizerIFType] = None
    dataset_batch_generator: PydanticDatasetBatchGeneratorIFType


class TrainStepMetrics(Enum):
    FORWARD_PASS_TIME_s = "forward_pass_time_s"
    BACKWARD_PASS_TIME_s = "backward_pass_time_s"
    OPTIMIZER_STEP_TIME_s = "optimizer_step_time_s"
    PEAK_MEMORY_MB = "peak_memory_MB"


class TrainStepStatistics:
    def __init__(self, global_rank: int, local_rank: int, num_ranks: int):
        self._global_rank: int = global_rank
        self._local_rank: int = local_rank
        self._num_ranks: int = num_ranks
        self._measurements_dict: dict[TrainStepMetrics, float] = defaultdict(list)

    @property
    def num_ranks(self) -> int:
        return self._num_ranks

    def add_measurement(self, step: TrainStepMetrics, time: float):
        self._measurements_dict[step].append(time)

    def add_measurements(self, measurements: dict[TrainStepMetrics, float]):
        for key, value in measurements.items():
            self._measurements_dict[key].append(value)

    def get_mean_measurements_dict(
        self,
    ) -> dict[TrainStepMetrics, float]:
        return {key: np.mean(values) for key, values in self._measurements_dict.items()}

    def __repr__(self):
        mean_measurements = self.get_mean_measurements_dict()
        mean_measurements["TOTAL_STEP_TIME_s"] = (
            mean_measurements[TrainStepMetrics.FORWARD_PASS_TIME_s]
            + mean_measurements[TrainStepMetrics.BACKWARD_PASS_TIME_s]
            + mean_measurements[TrainStepMetrics.OPTIMIZER_STEP_TIME_s]
        )
        lines = ["\nStep statistics global rank {} (local rank: {}):".format(self._global_rank, self._local_rank)]
        lines.append(f"{'Measurement':<30} {'Value':>10}")
        lines.extend([f"{k.name:<30} {v:>10.3f}" for k, v in mean_measurements.items()])
        return "\n".join(lines)


@dataclass
class Result:
    @dataclass
    class Measurement:
        peak_memory: float
        forward_time: float
        backward_time: float
        step_time: float

    @dataclass
    class EnvInfo:
        local_rank: int
        global_rank: int
        num_ranks: int
        hostname: str

        @staticmethod
        def from_env() -> "Result.EnvInfo":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            global_rank = int(os.environ.get("RANK", 0))  # torchrun uses RANK for global rank
            num_ranks = int(os.environ.get("WORLD_SIZE", 0))
            hostname = socket.gethostname()
            return Result.EnvInfo(
                local_rank=local_rank, global_rank=global_rank, num_ranks=num_ranks, hostname=hostname
            )

    grid_search_config: dict[str, ConfigValue]
    env_info: EnvInfo
    measurement: Measurement
    error: str = ""


class ModalitiesProfiler:
    @staticmethod
    def get_train_step_statistics(
        config_file_path: Path,
        experiment_folder_path: Path,
        num_warmup_steps: int,
        num_measurement_steps: int,
    ):
        error = ""
        try:
            with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
                experiment_folder_path = Path(
                    get_synced_string(string_to_be_synced=str(experiment_folder_path), from_rank=0)
                )

                step_statistics = ModalitiesProfiler.get_train_step_statistics_impl(
                    config_file_path=config_file_path,
                    num_warmup_steps=num_warmup_steps,
                    num_measurement_steps=num_measurement_steps,
                )
                mean_statistics = step_statistics.get_mean_measurements_dict()
        except Exception as e:
            error = str(e)
            mean_statistics = defaultdict(lambda: -1)

        with open(config_file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        result = Result(
            grid_search_config=config_dict["settings"]["benchmark"],
            measurement=Result.Measurement(
                peak_memory=mean_statistics[TrainStepMetrics.PEAK_MEMORY_MB],
                forward_time=mean_statistics[TrainStepMetrics.FORWARD_PASS_TIME_s],
                backward_time=mean_statistics[TrainStepMetrics.BACKWARD_PASS_TIME_s],
                step_time=mean_statistics[TrainStepMetrics.OPTIMIZER_STEP_TIME_s],
            ),
            env_info=Result.EnvInfo.from_env(),
            error=error,
        )
        # write results to json on all ranks
        current_rank = int(os.environ["RANK"])
        hash = hashlib.sha256(str(config_dict).encode()).hexdigest()[:8]

        result_file_path = experiment_folder_path / f"{hash}_{current_rank}.json"
        # create folder if not exists
        result_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file_path, "w") as f:
            json.dump(asdict(result), f, indent=4)
        return result

    @staticmethod
    def get_train_step_statistics_impl(
        config_file_path: Path,
        num_warmup_steps: int,
        num_measurement_steps: int,
    ) -> TrainStepStatistics:
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        main_obj = Main(config_file_path)
        components = main_obj.build_components(components_model_type=InstantiationModel)
        model = components.initialized_model
        loss_fun: Loss = components.loss_fn
        optimizer: Optional[torch.optim.Optimizer] = components.optimizer
        batch_generator: DatasetBatchGeneratorIF = components.dataset_batch_generator
        statistics = TrainStepStatistics(
            global_rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            num_ranks=torch.distributed.get_world_size(),
        )
        for _ in range(num_warmup_steps):
            ModalitiesProfiler._run_train_step(
                model=model,
                batch_generator=batch_generator,
                loss_fun=loss_fun,
                optimizer=optimizer,
            )
        for _ in range(num_measurement_steps):
            measurements_dict = ModalitiesProfiler._run_train_step(
                model=model,
                batch_generator=batch_generator,
                loss_fun=loss_fun,
                optimizer=optimizer,
            )
            statistics.add_measurements(measurements=measurements_dict)

        return statistics

    @staticmethod
    def get_forward_pass_statistics(
        config_file_path: Path,
        batch_generator: Callable[[], DatasetBatch],
        num_measurement_steps: int,
        profile_context_manager: torch.profiler.profile,
    ) -> TrainStepStatistics:
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            main_obj = Main(config_file_path)
            components = main_obj.build_components(components_model_type=InstantiationModel)
            model = components.initialized_model
            loss_fun: Loss = components.loss_fn
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            with profile_context_manager as profiler:
                for _ in range(num_measurement_steps):
                    batch = batch_generator()
                    batch.to(device=device)
                    torch.distributed.barrier()
                    ModalitiesProfiler._run_forward_pass(
                        model=model,
                        batch=batch,
                        loss_fun=loss_fun,
                    )
                    profiler.step()

    @staticmethod
    def _run_train_step(
        model: FSDPX,
        batch_generator: DatasetBatchGeneratorIF,
        loss_fun: Callable,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict[TrainStepMetrics, float]:
        device = torch.device(f"cuda:{int(os.environ['RANK'])}")
        # generate batch
        batch = batch_generator.get_dataset_batch()
        batch.to(device=device)

        # forward pass
        torch.cuda.reset_peak_memory_stats(device)
        start_forward = time.time()
        predictions = model(batch.samples)
        forward_time = time.time() - start_forward

        result_batch = InferenceResultBatch(targets=batch.targets, predictions=predictions)
        loss = loss_fun(result_batch)

        # backward pass
        start_backward = time.time()
        loss.backward()
        backward_time = time.time() - start_backward

        # optimizer step
        if optimizer is not None:
            start_step = time.time()
            optimizer.step()
            optimizer.zero_grad()
            step_time = time.time() - start_step

        # calculate the peak memory
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # in MB
        batch_size = batch.samples["input_ids"].shape[0]
        return {
            TrainStepMetrics.FORWARD_PASS_TIME_s: forward_time / batch_size,  # per sample
            TrainStepMetrics.BACKWARD_PASS_TIME_s: backward_time / batch_size,
            TrainStepMetrics.OPTIMIZER_STEP_TIME_s: step_time / batch_size,
            TrainStepMetrics.PEAK_MEMORY_MB: peak_memory,
        }

    @staticmethod
    def _run_forward_pass(model: FSDPX, batch: DatasetBatch, loss_fun: Callable):
        predictions = model(batch.samples)
        result_batch = InferenceResultBatch(targets=batch.targets, predictions=predictions)
        loss_fun(result_batch)


def flatten_results(results: list[Result]) -> list[dict]:
    flat_results = []
    for r in results:
        flat = {
            **{
                config_value.name: config_value.value for _, config_value in r.grid_search_config.items()
            },  # flatten grid_search_config
            "peak_memory": r.peak_memory,
            "forward_time": r.forward_time,
            "backward_time": r.backward_time,
            "step_time": r.step_time,
            "error": r.error,
        }
        flat_results.append(flat)
    return flat_results
