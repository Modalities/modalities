import os
from pathlib import Path

import pytest
import torch.multiprocessing as mp
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydanctic_if_types import PydanticFSDP1ModuleType, PydanticFSDP2ModuleType
from modalities.models.gpt2.gpt2_model import GPT2Block
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv

working_dir = Path(os.path.dirname(__file__))


class ActivationCheckpointingInstantiationModel(BaseModel):
    activation_checkpointed_model: PydanticFSDP1ModuleType | PydanticFSDP2ModuleType


@pytest.mark.parametrize(
    "rdvz_port, world_size, relative_config_path",
    [
        (22310, 2, "config_activation_checkpointing_fsdp1.yaml"),
        (22311, 2, "config_activation_checkpointing_fsdp2.yaml"),
    ],
)
def test_activation_checkpointing(world_size: int, rdvz_port: int, relative_config_path: str):
    mp.spawn(
        _test_activation_checkpointing_thread,
        args=(rdvz_port, world_size, relative_config_path),
        nprocs=world_size,
        join=True,
    )


def _test_activation_checkpointing_thread(process_id: int, rdvz_port: int, world_size: int, relative_config_path: str):
    working_dir = Path(os.path.dirname(__file__))
    config_file_path = working_dir / relative_config_path

    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=world_size,
        rdvz_port=rdvz_port,
    ):
        main = Main(config_file_path)
        components = main.build_components(components_model_type=ActivationCheckpointingInstantiationModel)
        modules = dict(components.activation_checkpointed_model.named_modules())

        # make sure that we have two modules that are wrapped with activation checkpointing
        # i.e., the two GPT2Block modules defined in the config
        ac_modules = [(module_name, module) for module_name, module in modules.items() if isinstance(module, GPT2Block)]
        assert len(ac_modules) == 2
        assert all([module_name.endswith("_checkpoint_wrapped_module") for module_name, _ in ac_modules])
        # make sure that there are no other modules that are wrapped with activation checkpointing
        assert sum([module_name.endswith("_checkpoint_wrapped_module") for module_name in modules.keys()]) == len(
            ac_modules
        )
