import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from modalities.config.config import ProcessGroupBackendType
from modalities.models.model_factory import ModelFactory
from modalities.running_env.env_utils import FSDP2MixedPrecisionSettings, PyTorchDtypes
from modalities.running_env.fsdp.device_mesh import get_device_mesh
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv
from tests.utility import find_free_port


def _run_cp_optimizer_update(process_id: int, world_size: int, port: int) -> None:
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=world_size,
        rdvz_port=port,
    ):
        torch.cuda.set_device(process_id)
        device_mesh = get_device_mesh(
            device_type="cuda",
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=world_size,
            enable_loss_parallel=False,
            world_size=world_size,
        )
        model = nn.Linear(1, 1, bias=False, device="cuda")
        nn.init.ones_(model.weight)
        model = ModelFactory.get_fsdp2_wrapped_model(
            model=model,
            block_names=[],
            device_mesh=device_mesh,
            mixed_precision_settings=FSDP2MixedPrecisionSettings(
                param_dtype=PyTorchDtypes.FP_32,
                reduce_dtype=PyTorchDtypes.FP_32,
            ),
            reshard_after_forward=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        local_input = torch.tensor([[float(process_id + 1)]], device="cuda")
        model(local_input).sum().backward()
        optimizer.step()

        local_weight = model.weight.to_local().detach().float().reshape(1)
        expected = torch.tensor([0.85], device="cuda")
        torch.testing.assert_close(local_weight, expected)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test requires 2 GPUs.")
def test_cp_ranks_apply_identical_optimizer_update() -> None:
    world_size = 2
    mp.spawn(
        _run_cp_optimizer_update,
        args=(world_size, find_free_port()),
        nprocs=world_size,
        join=True,
    )
