import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import tqdm
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.batch import DatasetBatch, InferenceResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydanctic_if_types import PydanticFSDP2ModuleType, PydanticLossIFType, PydanticOptimizerIFType
from modalities.loss_functions import Loss
from modalities.running_env.cuda_env import CudaEnv
from modalities.utils.typing_utils import FSDPX


@dataclass
class Measurement:
    peak_memory: list[int]
    forward_time: list[float]
    backward_time: list[float]
    step_time: list[float]

    batch_size: int
    local_rank: int
    num_ranks: int

    def get_statistics(self) -> str:
        """
        Returns the statistics of the current process group.
        """
        statistics = f"""
            Rank: {self.local_rank}
            Batch size: {self.batch_size}
            Number of ranks: {self.num_ranks}
            Peak memory: {np.mean(self.peak_memory) / 1024 ** 2} MB
            Forward time: {np.mean(self.forward_time)} seconds
            Backward time: {np.mean(self.backward_time)} seconds
            Step time: {np.mean(self.step_time)} seconds
            Total time: {np.mean(self.forward_time) + np.mean(self.backward_time) + np.mean(self.step_time)} seconds
            """
        return statistics


class InstantiationModel(BaseModel):
    initialized_model: PydanticFSDP2ModuleType
    loss_fn: PydanticLossIFType
    optimizer: Optional[PydanticOptimizerIFType] = None


def get_batch(vocab_size: int, batch_size: int, sequence_length: int) -> DatasetBatch:
    batch = DatasetBatch(
        samples={"input_ids": torch.randint(0, vocab_size, (batch_size, sequence_length))},
        targets={"target_ids": torch.randint(0, 5000, (batch_size, sequence_length))},
    )
    return batch


def get_peak_memory(
    model: FSDPX,
    batch_generator: Callable[[], DatasetBatch],
    loss_fun: Callable,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[int, float, float, float]:
    # torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    # generate batch
    batch = batch_generator()
    batch.to(device=device)
    torch.distributed.barrier()

    # forward pass
    torch.cuda.synchronize()
    start_forward = time.time()
    predictions = model(batch.samples)
    torch.cuda.synchronize()
    forward_time = time.time() - start_forward

    result_batch = InferenceResultBatch(targets=batch.targets, predictions=predictions)
    loss = loss_fun(result_batch)

    # backward pass
    torch.cuda.synchronize()
    start_backward = time.time()
    loss.backward()
    torch.cuda.synchronize()
    backward_time = time.time() - start_backward

    # optimizer step
    if optimizer is not None:
        torch.cuda.synchronize()
        start_step = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        step_time = time.time() - start_step
    peak_memory = torch.cuda.max_memory_allocated()

    return peak_memory, forward_time, backward_time, step_time


def benchmark_activation_checkpoint_memory_saving(
    config_file_path: Path,
    batch_generator: Callable[[], DatasetBatch],
    num_measurements: int,
    num_warmups: int,
    batch_size: int,
):
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main_obj = Main(config_file_path)
        components = main_obj.build_components(components_model_type=InstantiationModel)
        model = components.initialized_model
        loss_fn: Loss = components.loss_fn
        optimizer: Optional[torch.optim.Optimizer] = components.optimizer
        # optimizer = None

        measurement = Measurement(
            peak_memory=[],
            forward_time=[],
            backward_time=[],
            step_time=[],
            batch_size=batch_size,
            local_rank=int(os.environ["LOCAL_RANK"]),
            num_ranks=torch.distributed.get_world_size(),
        )
        global_rank = int(os.environ["RANK"])
        if global_rank == 0:
            warmup_iterator = tqdm.tqdm(range(num_warmups), desc="Benchmarking", unit="iteration")
            measurment_iterator = tqdm.tqdm(range(num_measurements), desc="Benchmarking", unit="iteration")
        else:
            warmup_iterator = range(num_warmups)
            measurment_iterator = range(num_measurements)

        for _ in warmup_iterator:
            get_peak_memory(model=model, batch_generator=batch_generator, loss_fun=loss_fn, optimizer=optimizer)
        for _ in measurment_iterator:
            peak_memory, forward_time, backward_time, step_time = get_peak_memory(
                model=model,
                batch_generator=batch_generator,
                loss_fun=loss_fn,
                optimizer=optimizer,
            )
            measurement.peak_memory.append(peak_memory)
            measurement.forward_time.append(forward_time)
            measurement.backward_time.append(backward_time)
            measurement.step_time.append(step_time)
        torch.distributed.barrier()

        print(measurement.get_statistics())


if __name__ == "__main__":
    config_file_path = Path(
        "/raid/s3/opengptx/max_lue/repositories/modalities/tests/training/config_activation_checkpointing_fsdp2_benchmark.yaml"
    )
    num_measurements = 20
    num_warmups = 3
    batch_size = 16
    sequence_length = 4096
    vocab_size = 5000

    batch_generator = partial(get_batch, vocab_size=vocab_size, batch_size=batch_size, sequence_length=sequence_length)

    benchmark_activation_checkpoint_memory_saving(
        config_file_path=config_file_path,
        num_measurements=num_measurements,
        num_warmups=num_warmups,
        batch_generator=batch_generator,
        batch_size=batch_size,
    )
