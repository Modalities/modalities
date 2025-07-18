from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from pydantic import BaseModel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.tensor import DTensor, Replicate

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticDeviceMeshIFType, PydanticFSDP2ModuleType
from modalities.models.gpt2.gpt2_model import TransformerMLP
from modalities.models.model import SwiGLU
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


def patch_config_file(original_config_path: Path, activation_type: str, tmp_dir: Path) -> Path:
    """Patches the original configuration file to set a custom activation type."""
    with original_config_path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config_dict["model_raw"]["config"]["activation_type"] = activation_type

    tmp_file_path = tmp_dir / original_config_path.name
    with tmp_file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    return tmp_file_path


@pytest.fixture
def tmp_config_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("patched_configs")


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="This test requires exactly 4 GPUs",
)
class TestTensorParallelism:
    def _get_components(self, config_file_path: Path) -> Tuple[FSDP2, DeviceMesh]:
        class ComponentsInstantiationModel(BaseModel):
            model: PydanticFSDP2ModuleType
            device_mesh: PydanticDeviceMeshIFType

        main_obj = Main(config_file_path)
        components: ComponentsInstantiationModel = main_obj.build_components(
            components_model_type=ComponentsInstantiationModel
        )
        return components.model, components.device_mesh

    @pytest.mark.parametrize(
        "activation_type, fsdp2_config_path, tp_config_path, port",
        [
            (
                "gelu",
                Path("tests/fsdp2_parallelization/tp_test_configs/fsdp2_config.yaml"),
                Path("tests/fsdp2_parallelization/tp_test_configs/tp_config.yaml"),
                22755,
            ),
            (
                "swiglu",
                Path("tests/fsdp2_parallelization/tp_test_configs/fsdp2_config.yaml"),
                Path("tests/fsdp2_parallelization/tp_test_configs/tp_config.yaml"),
                22756,
            ),
        ],
    )
    def test_tp_sharding(
        self,
        activation_type: str,
        fsdp2_config_path: Path,
        tp_config_path: Path,
        tmp_config_dir: Path,
        port: int,
    ):
        world_size = 4
        mp.spawn(
            self._test_tp_sharding_impl,
            args=(activation_type, fsdp2_config_path, tp_config_path, world_size, tmp_config_dir, port),
            nprocs=world_size,
            join=True,
        )

    def _test_tp_sharding_impl(
        self,
        process_id: int,
        activation_type: str,
        fsdp2_config_path: Path,
        tp_config_path: Path,
        world_size: int,
        tmp_config_dir: Path,
        port: int,
    ):
        """Runs the sharding test logic for a single process in the distributed setup."""
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=port,
        ):
            # Seed before FSDP2 instantiation
            torch.manual_seed(42)
            fsdp2_path = patch_config_file(fsdp2_config_path, activation_type, tmp_config_dir)
            fsdp2_model, fsdp2_mesh = self._get_components(fsdp2_path)

            # Seed again before TP instantiation to match
            torch.manual_seed(42)
            tp_path = patch_config_file(tp_config_path, activation_type, tmp_config_dir)
            tp_model, tp_mesh = self._get_components(tp_path)

            # Ensure models use the correct MLP
            if activation_type == "gelu":
                assert isinstance(fsdp2_model.transformer.h[0].mlp, TransformerMLP)
                assert isinstance(tp_model.transformer.h[0].mlp, TransformerMLP)
            elif activation_type == "swiglu":
                assert isinstance(fsdp2_model.transformer.h[0].mlp, SwiGLU)
                assert isinstance(tp_model.transformer.h[0].mlp, SwiGLU)

            # Ensure models are sharded correctly
            assert "tp" in tp_model.transformer.wte.weight.device_mesh.mesh_dim_names
            assert "tp" not in fsdp2_model.transformer.wte.weight.device_mesh.mesh_dim_names

            # Compare weights
            mismatches = self._compare_model_state_dicts(fsdp2_model, tp_model, fsdp2_mesh, tp_mesh)
            assert not mismatches, f"Mismatch in model parameters: {mismatches}"

            # Compare outputs
            vocab_size = 50304
            sequence_length = 128
            batch_size = 2
            input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
            input_dict = {"input_ids": input_ids}

            out_fsdp2 = fsdp2_model(input_dict)["logits"].float()
            out_tp = tp_model(input_dict)["logits"].float()

            assert out_fsdp2.shape == out_tp.shape, "Output shapes do not match"
            assert torch.allclose(out_fsdp2, out_tp, atol=1e-6, rtol=1e-5), "Outputs do not match"

    @staticmethod
    def _compare_model_state_dicts(
        fsdp2_model: nn.Module,
        tp_model: nn.Module,
        fsdp2_mesh: DeviceMesh,
        tp_mesh: DeviceMesh,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> list[str]:
        """Returns a list of parameter names where model weights differ beyond tolerance."""
        mismatches = []

        def all_named_tensors(model: nn.Module):
            yield from model.named_parameters()
            yield from model.named_buffers()

        fsdp2_tensors = dict(all_named_tensors(fsdp2_model))
        tp_tensors = dict(all_named_tensors(tp_model))

        assert fsdp2_tensors.keys() == tp_tensors.keys(), "Model structures differ"

        for name in fsdp2_tensors:
            a, b = fsdp2_tensors[name], tp_tensors[name]

            a_mat = a.redistribute(fsdp2_mesh, [Replicate()]).to_local() if isinstance(a, DTensor) else a
            b_mat = b.redistribute(tp_mesh, [Replicate(), Replicate()]).to_local() if isinstance(b, DTensor) else b

            if not torch.allclose(a_mat, b_mat, atol=atol, rtol=rtol):
                mismatches.append(name)

        return mismatches
