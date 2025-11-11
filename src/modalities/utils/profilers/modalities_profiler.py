import os
from pathlib import Path
from typing import Callable, Optional

import torch
from pydantic import BaseModel
from torch.profiler import ProfilerActivity, profile, schedule

from modalities.__main__ import Main
from modalities.batch import DatasetBatch, InferenceResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import (
    PydanticDatasetBatchGeneratorIFType,
    PydanticFSDP2ModuleType,
    PydanticLossIFType,
    PydanticOptimizerIFType,
)
from modalities.loss_functions import Loss
from modalities.running_env.cuda_env import CudaEnv
from modalities.utils.profilers.batch_generator import DatasetBatchGeneratorIF
from modalities.utils.typing_utils import FSDP2


class InstantiationModel(BaseModel):
    initialized_model: PydanticFSDP2ModuleType
    loss_fn: PydanticLossIFType
    optimizer: Optional[PydanticOptimizerIFType] = None
    dataset_batch_generator: PydanticDatasetBatchGeneratorIFType


class ModalitiesProfiler:
    @staticmethod
    def get_forward_pass_profiling(
        config_file_path: Path,
        num_measurement_steps: int,
        profile_context_manager: torch.profiler.profile,
    ):
        def _run_forward_pass(model: FSDP2, batch: DatasetBatch, loss_fun: Optional[Callable] = None) -> None:
            predictions = model(batch.samples)
            result_batch = InferenceResultBatch(targets=batch.targets, predictions=predictions)
            if loss_fun is not None:
                loss_fun(result_batch)

        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            main_obj = Main(config_file_path)
            components: InstantiationModel = main_obj.build_components(components_model_type=InstantiationModel)
            model = components.initialized_model
            loss_fun: Loss = components.loss_fn
            dataset_batch_generator: DatasetBatchGeneratorIF = components.dataset_batch_generator
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            with profile_context_manager as profiler:
                for _ in range(num_measurement_steps):
                    batch = dataset_batch_generator.get_dataset_batch()
                    batch.to(device=device)
                    _run_forward_pass(
                        model=model,
                        batch=batch,
                        loss_fun=loss_fun,
                    )
                    profiler.step()

            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    print("Saving profiling results...")
                    profiler_context_manager.export_chrome_trace(output_path.as_posix())
                    print(profiler_context_manager.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    print("Profiling complete.")


if __name__ == "__main__":
    config_path = Path(
        "/raid/s3/opengptx/max_lue/repositories/modalities/config_files/profiling/8B_profiling_config.yaml"
    )
    output_path = Path("/raid/s3/opengptx/max_lue/repositories/modalities/outputs/profiler_trace.json")

    profiler_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    num_measurements = 3
    wait = 20
    warmup = 20
    total = wait + warmup + num_measurements

    profiler_context_manager = profile(
        activities=profiler_activities,
        schedule=schedule(wait=wait, warmup=warmup, active=num_measurements),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=True,
        with_modules=True,
    )

    ModalitiesProfiler.get_forward_pass_profiling(
        config_file_path=config_path,
        num_measurement_steps=total,
        profile_context_manager=profiler_context_manager,
    )
