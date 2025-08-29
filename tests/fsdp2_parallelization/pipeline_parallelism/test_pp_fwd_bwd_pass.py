import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticFSDP2ModuleType, PydanticPipelineType
from modalities.models.parallelism.pipeline_parallelism import Pipeline
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@pytest.fixture
def temp_file_path() -> Path:
    # Create a NamedTemporaryFile that persists after closing (delete=False)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        file_path = tf.name
    try:
        yield Path(file_path)
    finally:
        # Clean up the file after the test
        if os.path.exists(file_path):
            os.remove(file_path)


class ComponentsInstantiationModel(BaseModel):
    initialized_model: PydanticFSDP2ModuleType
    scheduled_pipeline: PydanticPipelineType


@pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="This test requires 8 GPUs",
)
class TestPipelineParallelism:
    def _get_tmp_sharding_config_path(
        self, sharding_degree: int, tp_degree: int, pp_degree: int, temp_file_path: Path
    ) -> Path:
        working_dir = Path(os.path.dirname(__file__))
        config_file_path = working_dir / "configs/config_lorem_ipsum_long_fsdp2_pp_fwd_bwd_pass.yaml"

        with open(config_file_path, "r") as file:
            config_string = file.read()
            config_dict = yaml.safe_load(config_string)
            config_dict["device_mesh"]["config"]["data_parallel_shard_degree"] = sharding_degree
            config_dict["device_mesh"]["config"]["tensor_parallel_degree"] = tp_degree
            config_dict["device_mesh"]["config"]["pipeline_parallel_degree"] = pp_degree

        # save to temporary file
        with open(temp_file_path, "w") as file:
            yaml.dump(config_dict, file)

        return temp_file_path

    def _get_components(self, config_file_path: Path) -> ComponentsInstantiationModel:
        main_obj = Main(config_file_path)
        components: ComponentsInstantiationModel = main_obj.build_components(
            components_model_type=ComponentsInstantiationModel
        )
        return components

    @pytest.mark.parametrize(
        "sharding_degree, tp_degree, pp_degree, world_size",
        [
            (2, 1, 2, 4),
            # (2, 1, 4, 8),
            # (2, 2, 2, 8), # TODO need to support this case
        ],
    )
    def test_pp(self, sharding_degree: int, tp_degree: int, pp_degree: int, world_size: int, temp_file_path: Path):
        tmp_sharding_config_path = self._get_tmp_sharding_config_path(
            sharding_degree=sharding_degree,
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            temp_file_path=temp_file_path,
        )
        mp.spawn(
            self._test_pp_impl,
            args=(world_size, tmp_sharding_config_path),
            nprocs=world_size,
            join=True,
        )

    def _test_pp_impl(
        self,
        process_id: int,
        world_size: int,
        gpt2_model_config_path: Path,
    ):
        # wraps the actual test function to be able to run it in a distributed  multiprocessing setup
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=22356,
        ):
            components = self._get_components(gpt2_model_config_path)
            scheduled_pipeline = components.scheduled_pipeline
            vocab_size = 50304
            sequence_length = 256
            batch_size = 4
            sequences = torch.randint(0, vocab_size, (batch_size, sequence_length))
            targets = sequences[:, 1:].contiguous()
            inputs = sequences[:, :-1].contiguous()
            self._forward_step(scheduled_pipeline, inputs, targets)

    def _forward_step(self, scheduled_pipeline: Pipeline, inputs: torch.Tensor, targets: torch.Tensor):
        """Runs a forward step on the model."""
        pp_schedule = scheduled_pipeline.pp_schedule
        targets, losses = (targets, []) if scheduled_pipeline.is_last_pp_stage else (None, None)
        if scheduled_pipeline.is_first_pp_stage:  # first stage
            pp_schedule.step(inputs, target=targets, losses=losses, input_batch=inputs)
        else:  # non-first stage
            pp_schedule.step(target=targets, losses=losses, input_batch=inputs)

        # accumulate losses across pipeline microbatches
        # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
        (
            torch.mean(torch.stack(losses)).to(self.device)
            if self.pp_has_last_stage
            else torch.tensor([-1.0], device=self.device)
        )

        # return output
