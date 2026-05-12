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
from torch.distributed.tensor import DTensor

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticDeviceMeshIFType, PydanticFSDP2ModuleType
from modalities.models.gpt2.gpt2_model import TransformerMLP, context_parallel, context_parallel_unshard
from modalities.models.model import SwiGLU
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv
from tests.utility import find_free_port


def patch_config_file(original_config_path: Path, activation_type: str, tmp_dir: Path, tag: str) -> Path:
    """Patch config to set activation type and write rank-unique temporary config files."""
    with original_config_path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config_dict["model_raw"]["config"]["activation_type"] = activation_type

    tmp_file_path = tmp_dir / f"{original_config_path.stem}_{tag}{original_config_path.suffix}"
    with tmp_file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    return tmp_file_path


@pytest.fixture
def tmp_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("patched_cp_configs")


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="This test requires exactly 4 GPUs",
)
@pytest.mark.skipif(
    context_parallel is None or context_parallel_unshard is None,
    reason="This test requires a PyTorch build with torch.distributed context parallel APIs",
)
class TestContextParallelism:
    def _get_components(self, config_file_path: Path, tmp_path: Path) -> Tuple[FSDP2, DeviceMesh]:
        class ComponentsInstantiationModel(BaseModel):
            model: PydanticFSDP2ModuleType
            device_mesh: PydanticDeviceMeshIFType

        main_obj = Main(config_file_path, experiments_root_path=tmp_path)
        components: ComponentsInstantiationModel = main_obj.build_components(
            components_model_type=ComponentsInstantiationModel
        )
        return components.model, components.device_mesh

    @pytest.mark.parametrize(
        "activation_type, fsdp2_config_path, cp_config_path",
        [
            (
                "gelu",
                Path("tests/fsdp2_parallelization/cp_test_configs/fsdp2_config.yaml"),
                Path("tests/fsdp2_parallelization/cp_test_configs/cp_config.yaml"),
            ),
            (
                "swiglu",
                Path("tests/fsdp2_parallelization/cp_test_configs/fsdp2_config.yaml"),
                Path("tests/fsdp2_parallelization/cp_test_configs/cp_config.yaml"),
            ),
        ],
    )
    def test_cp_runtime_parity(
        self, activation_type: str, fsdp2_config_path: Path, cp_config_path: Path, tmp_config_dir: Path, tmp_path: Path
    ):
        world_size = 4
        port = find_free_port()
        mp.spawn(
            self._test_cp_runtime_parity_impl,
            args=(activation_type, fsdp2_config_path, cp_config_path, world_size, tmp_config_dir, port, tmp_path),
            nprocs=world_size,
            join=True,
        )

    def _test_cp_runtime_parity_impl(
        self,
        process_id: int,
        activation_type: str,
        fsdp2_config_path: Path,
        cp_config_path: Path,
        world_size: int,
        tmp_config_dir: Path,
        port: int,
        tmp_path: Path,
    ):
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=port,
        ):
            torch.manual_seed(42)
            fsdp2_path = patch_config_file(
                original_config_path=fsdp2_config_path,
                activation_type=activation_type,
                tmp_dir=tmp_config_dir,
                tag=f"rank{process_id}_fsdp2",
            )
            fsdp2_model, fsdp2_mesh = self._get_components(fsdp2_path, tmp_path)

            torch.manual_seed(42)
            cp_path = patch_config_file(
                original_config_path=cp_config_path,
                activation_type=activation_type,
                tmp_dir=tmp_config_dir,
                tag=f"rank{process_id}_cp",
            )
            cp_model, cp_mesh = self._get_components(cp_path, tmp_path)

            if activation_type == "gelu":
                assert isinstance(fsdp2_model.transformer.h["0"].mlp, TransformerMLP)
                assert isinstance(cp_model.transformer.h["0"].mlp, TransformerMLP)
            elif activation_type == "swiglu":
                assert isinstance(fsdp2_model.transformer.h["0"].mlp, SwiGLU)
                assert isinstance(cp_model.transformer.h["0"].mlp, SwiGLU)

            assert cp_model.transformer.h["0"].attn.context_parallel_mesh is not None
            assert "cp" in cp_model.transformer.h["0"].attn.context_parallel_mesh.mesh_dim_names
            assert fsdp2_model.transformer.h["0"].attn.context_parallel_mesh is None

            mismatches = self._compare_model_state_dicts(fsdp2_model, cp_model, fsdp2_mesh, cp_mesh)
            assert not mismatches, f"Mismatch in model parameters: {mismatches}"

            vocab_size = 50304
            sequence_length = 128
            batch_size = 2
            input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
            input_dict = {"input_ids": input_ids}

            out_fsdp2 = fsdp2_model(input_dict)["logits"].float()
            out_cp = cp_model(input_dict)["logits"].float()

            assert out_fsdp2.shape == out_cp.shape, "Output shapes do not match"
            assert torch.allclose(out_fsdp2, out_cp, atol=1e-6, rtol=1e-5), "Outputs do not match"

            fsdp2_model.zero_grad(set_to_none=True)
            cp_model.zero_grad(set_to_none=True)

            loss_fsdp2 = out_fsdp2.square().mean()
            loss_cp = out_cp.square().mean()

            loss_fsdp2.backward()
            loss_cp.backward()

            grad_mismatches = self._compare_model_grads(fsdp2_model, cp_model, atol=1e-5, rtol=1e-4)
            assert not grad_mismatches, f"Mismatch in parameter gradients: {grad_mismatches}"

    @staticmethod
    def _compare_model_state_dicts(
        fsdp2_model: nn.Module,
        cp_model: nn.Module,
        fsdp2_mesh: DeviceMesh,
        cp_mesh: DeviceMesh,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> list[str]:
        """Return tensor names where model weights differ beyond tolerance."""
        mismatches = []

        def all_named_tensors(model: nn.Module):
            yield from model.named_parameters()
            yield from model.named_buffers()

        fsdp2_tensors = dict(all_named_tensors(fsdp2_model))
        cp_tensors = dict(all_named_tensors(cp_model))

        assert fsdp2_tensors.keys() == cp_tensors.keys(), "Model structures differ"

        for name in fsdp2_tensors:
            a, b = fsdp2_tensors[name], cp_tensors[name]

            a_mat = a.full_tensor() if isinstance(a, DTensor) else a
            b_mat = b.full_tensor() if isinstance(b, DTensor) else b

            if not torch.allclose(a_mat, b_mat, atol=atol, rtol=rtol):
                mismatches.append(name)

        return mismatches

    @staticmethod
    def _compare_model_grads(
        fsdp2_model: nn.Module,
        cp_model: nn.Module,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> list[str]:
        """Return parameter names where gradients differ beyond tolerance."""
        mismatches = []

        fsdp2_params = dict(fsdp2_model.named_parameters())
        cp_params = dict(cp_model.named_parameters())
        assert fsdp2_params.keys() == cp_params.keys(), "Model parameter structures differ"

        for name in fsdp2_params:
            grad_a = fsdp2_params[name].grad
            grad_b = cp_params[name].grad

            if grad_a is None and grad_b is None:
                continue
            if grad_a is None or grad_b is None:
                mismatches.append(name)
                continue

            if isinstance(grad_a, DTensor):
                grad_a = grad_a.full_tensor()
            if isinstance(grad_b, DTensor):
                grad_b = grad_b.full_tensor()

            if not torch.allclose(grad_a.float(), grad_b.float(), atol=atol, rtol=rtol):
                mismatches.append(name)

        return mismatches
