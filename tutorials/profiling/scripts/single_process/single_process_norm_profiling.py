from pathlib import Path

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.profiler import record_function

from modalities.config.pydantic_if_types import PydanticDatasetBatchGeneratorIFType, PydanticPytorchModuleType
from modalities.utils.profilers.batch_generator import DatasetBatchGeneratorIF
from modalities.utils.profilers.modalities_profiler import CustomComponentRegisterable, ModalitiesProfilerStarter
from modalities.utils.profilers.steppable_components_if import SteppableComponentIF


class SteppableNormConfig(BaseModel):
    norm: PydanticPytorchModuleType
    dataset_batch_generator: PydanticDatasetBatchGeneratorIFType


class SteppableNorm(SteppableComponentIF):
    """A steppable component that applies Norm to batches from a dataset batch generator.
    Used for profiling or inspecting normalization performance.
    """

    def __init__(
        self, dataset_batch_generator: DatasetBatchGeneratorIF, norm: nn.Module, apply_compile: bool = False
    ) -> None:
        self.dataset_batch_generator = dataset_batch_generator
        self.norm = norm
        self.device = torch.device("cuda")
        self.norm.to(self.device)
        self.norm.to(torch.bfloat16)
        if apply_compile:
            self.norm = torch.compile(self.norm)

    def step(self) -> None:
        batch = self.dataset_batch_generator.get_dataset_batch()
        batch.to(device=self.device)
        with record_function("rms_norm_inference"):
            self.norm(batch.samples["input_ids"])


if __name__ == "__main__":
    cwd = Path(__file__).parent.resolve()
    config_path = cwd / Path("../../configs/single_process_rms_norm_profiling.yaml")
    experiment_root_path = Path("../../experiments/")

    custom_component_registerables = [
        CustomComponentRegisterable(
            component_key="steppable_component",
            variant_key="steppable_norm",
            custom_component=SteppableNorm,
            custom_config=SteppableNormConfig,
        )
    ]

    ModalitiesProfilerStarter.run_single_process(
        config_file_path=config_path,
        experiment_root_path=experiment_root_path,
        custom_component_registerables=custom_component_registerables,
    )
