import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.batch import InferenceResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import (
    PydanticDeviceMeshIFType,
    PydanticFSDP2ModuleType,
    PydanticLossIFType,
    PydanticPipelineType,
)
from modalities.models.parallelism.pipeline_parallelism import Pipeline
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_parallel_rank
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


class ComponentsInstantiationPPModel(BaseModel):
    scheduled_pipeline: PydanticPipelineType
    device_mesh: PydanticDeviceMeshIFType


class ComponentsInstantiationModel(BaseModel):
    fsdp_model: PydanticFSDP2ModuleType
    loss_fn: PydanticLossIFType
    device_mesh: PydanticDeviceMeshIFType


@pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="This test requires 8 GPUs",
)
class TestPipelineParallelism:
    @pytest.mark.parametrize(
        "fsdp_degree, tp_degree, pp_degree, world_size",
        [
            (2, 1, 2, 4),
            (2, 2, 2, 8),
        ],
    )
    def test_compare_pp_step_with_fsdp2_only_forward_backward_step(
        self, fsdp_degree: int, tp_degree: int, pp_degree: int, world_size: int, tmp_path: Path
    ):
        tmp_sharding_config_path = self._get_tmp_sharding_config_path(
            fsdp_degree=fsdp_degree, tp_degree=tp_degree, pp_degree=pp_degree, tmp_path=tmp_path
        )
        mp.spawn(
            self._test_pp_impl,
            args=(world_size, tmp_sharding_config_path),
            nprocs=world_size,
            join=True,
        )

    def _get_data(self, dp_rank: int):
        vocab_size = 50304
        sequence_length = 256
        batch_size = 4
        # since we compare runs with different dp_degrees 2 and 4, ensure that every second dp rank gets the same data
        torch.manual_seed(dp_rank % 2)
        sequences = torch.randint(0, vocab_size, (batch_size, sequence_length + 1))
        targets = sequences[:, 1:].contiguous()
        inputs = sequences[:, :-1].contiguous()
        return inputs, targets

    def _test_pp_impl(
        self,
        process_id: int,
        world_size: int,
        pp_model_config_path: Path,
    ):
        # wraps the actual test function to be able to run it in a distributed multiprocessing setup
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=22359,
        ):
            is_last_pp_stage, loss_pp = self._forward_step_with_pp(pp_model_config_path)
            fsdp2_loss = self._forward_step_without_pp()

            if is_last_pp_stage:
                assert torch.allclose(
                    loss_pp, fsdp2_loss, atol=1e-6, rtol=1e-5
                ), f"Losses do not match.\nLoss with PP: {loss_pp.item()}, Loss without PP: {fsdp2_loss.item()}"

    def _forward_step_with_pp(
        self,
        pp_model_config_path: Path,
    ) -> tuple[bool, torch.Tensor]:
        try:
            components = self._get_components(pp_model_config_path, use_pp=True)
            dp_rank = get_parallel_rank(components.device_mesh, ParallelismDegrees.DP_SHARD)
            inputs, targets = self._get_data(dp_rank=dp_rank)
            scheduled_pipeline = components.scheduled_pipeline
            loss_pp = self._forward_step(scheduled_pipeline, inputs, targets)
        except Exception as e:
            import traceback

            print(f"Exception in _forward_step_with_pp: {e}")
            traceback.print_exc()
            raise e
        return scheduled_pipeline.has_last_pp_stage, loss_pp

    def _forward_step(self, scheduled_pipeline: Pipeline, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Runs a forward step on the model."""
        pp_schedule = scheduled_pipeline.pp_schedule
        targets, losses = (targets, []) if scheduled_pipeline.has_last_pp_stage else (None, None)
        if scheduled_pipeline.has_first_pp_stage:
            pp_schedule.step(inputs, target=targets, losses=losses)
        else:
            pp_schedule.step(target=targets, losses=losses)

        # accumulate losses across pipeline microbatches
        return (
            torch.mean(torch.stack(losses)).to(losses[0].device)
            if scheduled_pipeline.has_last_pp_stage
            else torch.tensor([-1.0], device=inputs.device)
        )

    def _forward_step_without_pp(self) -> torch.Tensor:
        working_dir = Path(os.path.dirname(__file__))
        fsdp2_model_config_path = working_dir / "configs/config_lorem_ipsum_long_fsdp2_fwd_bwd_pass.yaml"
        components = self._get_components(fsdp2_model_config_path, use_pp=False)
        dp_rank = get_parallel_rank(components.device_mesh, ParallelismDegrees.DP_SHARD)
        inputs, targets = self._get_data(dp_rank=dp_rank)
        fsdp2_model = components.fsdp_model
        fsdp2_loss_fn = components.loss_fn

        input_dict = {"input_ids": inputs}
        fsdp2_out = fsdp2_model(input_dict)
        forward_batch = InferenceResultBatch(predictions=fsdp2_out, targets={fsdp2_loss_fn.target_key: targets})
        fsdp2_loss = fsdp2_loss_fn(forward_batch)
        return fsdp2_loss

    def _get_tmp_sharding_config_path(self, fsdp_degree: int, tp_degree: int, pp_degree: int, tmp_path: Path) -> Path:
        temp_file_path = tmp_path / "pp_sharding_config.yaml"
        working_dir = Path(os.path.dirname(__file__))
        if tp_degree > 1:
            config_file_path = working_dir / "configs/config_lorem_ipsum_long_fsdp2_pp_tp_fwd_bwd_pass.yaml"
        else:
            config_file_path = working_dir / "configs/config_lorem_ipsum_long_fsdp2_pp_fwd_bwd_pass.yaml"

        with open(config_file_path, "r") as file:
            config_string = file.read()
            config_dict = yaml.safe_load(config_string)
            config_dict["device_mesh"]["config"]["data_parallel_shard_degree"] = fsdp_degree
            config_dict["device_mesh"]["config"]["tensor_parallel_degree"] = tp_degree
            config_dict["device_mesh"]["config"]["pipeline_parallel_degree"] = pp_degree

        # save to temporary file
        with open(temp_file_path, "w") as file:
            yaml.dump(config_dict, file)

        return temp_file_path

    def _get_components(
        self, config_file_path: Path, use_pp: bool
    ) -> ComponentsInstantiationPPModel | ComponentsInstantiationModel:
        torch.manual_seed(42)
        main_obj = Main(config_file_path)
        components_model_type = ComponentsInstantiationPPModel if use_pp else ComponentsInstantiationModel
        components = main_obj.build_components(components_model_type=components_model_type)
        assert isinstance(components, components_model_type)
        return components
