import os
from pathlib import Path

# import debugpy
import pytest
import torch
import torch.cuda
from pydantic import BaseModel
from torch.distributed._tensor import DTensor

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydanctic_if_types import PydanticDeviceMeshIFType, PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.running_env.cuda_env import CudaEnv

rank = int(os.getenv("RANK", 0))

# debugpy.listen(5691 + rank)  # You can choose a different port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

working_dir = Path(os.path.dirname(__file__))


class InstantiationFSDPModel(BaseModel):
    model_wrapped: PydanticPytorchModuleType
    model_raw: PydanticPytorchModuleType
    device_mesh: PydanticDeviceMeshIFType


class InstantiationRawModel(BaseModel):
    model_raw: PydanticPytorchModuleType


def _gather_tensor(tensor: torch.Tensor):
    """
    Gathers a tensor from all shards to make it comparable with the non-sharded version.
    Assumes the tensor is on the CPU or will be moved to the CPU for gathering.
    """
    gathered_tensor = torch.empty_like(tensor)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    gathered_tensor.copy_(tensor)
    return gathered_tensor


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_fsdp_2_wrapped_model_weight_equivalence():
    # This test checks that the weights of the FSDP model are the same as the weights of the raw model.

    config_path = working_dir / "configs/fsdp2_model_config.yaml"
    main = Main(config_path=config_path)

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        torch.manual_seed(0)
        components_fsdp: InstantiationFSDPModel = main.build_components(components_model_type=InstantiationFSDPModel)
        torch.manual_seed(0)
        components_raw: InstantiationRawModel = main.build_components(components_model_type=InstantiationRawModel)

        model_raw: GPT2LLM = components_raw.model_raw
        fsdp_2_model = components_fsdp.model_wrapped

        for (name_raw, param_raw), (name_fsdp, param_fsdp) in zip(
            model_raw.named_parameters(), fsdp_2_model.named_parameters()
        ):
            assert name_raw == name_fsdp
            assert isinstance(param_fsdp, DTensor)

            shard_dim = param_fsdp.placements[0].dim
            mesh = param_fsdp.device_mesh.mesh
            rank = int(os.getenv("RANK"))
            local_tensor = param_fsdp._local_tensor

            assert local_tensor.device == torch.device(f"cuda:{rank}")

            # Since the weights of the fsdp model are sharded, we need to build an index to
            #  compare them with the raw model weights
            # (The raw model has a super set of the weights of the fsdp model on each rank)
            index = [slice(None)] * param_raw.ndim  # This creates a list of slices, one for each dimension
            num_elements = param_raw.shape[shard_dim] // len(mesh)
            index[shard_dim] = list(
                range(num_elements * rank, num_elements * rank + num_elements)
            )  # Replace the slice with the specific index at the n-th dimension

            assert torch.all(param_raw[index] == local_tensor.cpu())


# TODO: test hybrid sharding


def test_fsdp_2_wrapped_model_forward_pass_equivalence():
    config_path = working_dir / "configs/fsdp2_model_config.yaml"
    main = Main(config_path=config_path)

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        torch.manual_seed(0)
        components_fsdp: InstantiationFSDPModel = main.build_components(components_model_type=InstantiationFSDPModel)
        torch.manual_seed(0)
        components_raw: InstantiationRawModel = main.build_components(components_model_type=InstantiationRawModel)

        model_raw: GPT2LLM = components_raw.model_raw.bfloat16()
        fsdp_2_model = components_fsdp.model_wrapped

        batch_size = 4
        vocab_size = model_raw.transformer.wte.weight.shape[0]
        sequence_length = model_raw.sequence_length
        random_tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
        input_dict = {model_raw.sample_key: random_tensor}

        output_model_raw = model_raw(input_dict)[model_raw.prediction_key]
        output_model_fsdp_2 = fsdp_2_model(input_dict)[model_raw.prediction_key].cpu()

        # evaluation formula: |actual-expected| <= atol+rtol*|expected|
        torch.testing.assert_close(
            output_model_raw,
            output_model_fsdp_2,
            atol=0.017,  # default for bfloat16: 1e-5
            rtol=0,  # default for bfloat16: 0.016
        )
