from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pydantic import BaseModel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.tensor import DTensor, Replicate

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticDeviceMeshIFType, PydanticFSDP2ModuleType
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="This test requires exactly 4 GPUs",
)
class TestTensorParallelism:
    def _get_components(self, config_file_path: Path) -> Tuple[FSDP2, DeviceMesh]:
        class ComponentsInstantiationModel(BaseModel):
            fsdp_model: PydanticFSDP2ModuleType
            device_mesh: PydanticDeviceMeshIFType

        main_obj = Main(config_file_path)
        components: ComponentsInstantiationModel = main_obj.build_components(
            components_model_type=ComponentsInstantiationModel
        )
        return components.fsdp_model, components.device_mesh

    @pytest.mark.parametrize(
        "fsdp_config_path, tp_config_path",
        [
            (
                Path("tests/fsdp2_parallelization/tp_test_configs/fsdp_config.yaml"),
                Path("tests/fsdp2_parallelization/tp_test_configs/tp_config.yaml"),
            ),
        ],
    )
    def test_tp_sharding(self, fsdp_config_path: Path, tp_config_path: Path):
        world_size = 4
        mp.spawn(
            self._test_tp_sharding_impl,
            args=(fsdp_config_path, tp_config_path, world_size),
            nprocs=world_size,
            join=True,
        )

    def _test_tp_sharding_impl(self, process_id: int, fsdp2_config_path: Path, tp_config_path: Path, world_size: int):
        # wraps the actual test function to be able to run it in a distributed  multiprocessing setup
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=22490,
        ):
            torch.manual_seed(42)
            fsdp2_model, fsdp2_device_mesh = self._get_components(fsdp2_config_path)
            torch.manual_seed(42)
            tp_model, tp_device_mesh = self._get_components(tp_config_path)

            # Ensure that the FSDP2 and TP model have the same weights
            mismatches = TestTensorParallelism._compare_model_state_dicts(
                fsdp2_model, tp_model, fsdp2_device_mesh, tp_device_mesh
            )
            assert len(mismatches) == 0, f"Mismatch in model parameters: {mismatches}"

            # Ensure that the forward pass leads to the same output
            vocab_size = 50304
            sequence_length = 128
            batch_size = 2
            input_tensor = torch.randint(0, vocab_size, (batch_size, sequence_length))
            input_dict = {"input_ids": input_tensor}

            fsdp2_output = fsdp2_model(input_dict)["logits"].float()
            tp_output = tp_model(input_dict)["logits"].float()

            assert fsdp2_output.shape == tp_output.shape, "Output shapes do not match"
            assert torch.allclose(fsdp2_output, tp_output, atol=1e-6, rtol=1e-5), "Outputs do not match"

    @staticmethod
    def _compare_model_state_dicts(
        fsdp2_model: nn.Module,
        tp_model: nn.Module,
        fsdp2_device_mesh: DeviceMesh,
        tp_device_mesh: DeviceMesh,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> list[str]:
        mismatches = []

        def get_all_named_tensors(model):
            for name, param in model.named_parameters():
                yield name, param
            for name, buf in model.named_buffers():
                yield name, buf

        params_fsdp2 = dict(get_all_named_tensors(fsdp2_model))
        params_tp = dict(get_all_named_tensors(tp_model))

        assert params_fsdp2.keys() == params_tp.keys(), "Model structures differ"

        for name in params_fsdp2:
            t_fsdp2, t_tp = params_fsdp2[name], params_tp[name]

            # Redistribute DTensors to Replicate
            if isinstance(t_fsdp2, DTensor):
                t_a_materialized = t_fsdp2.redistribute(fsdp2_device_mesh, [Replicate()]).to_local()
            if isinstance(t_tp, DTensor):
                t_b_materialized = t_tp.redistribute(tp_device_mesh, [Replicate(), Replicate()]).to_local()

            if not torch.allclose(t_a_materialized, t_b_materialized, atol=atol, rtol=rtol):
                mismatches.append(name)

        return mismatches
