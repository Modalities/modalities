import os
from pathlib import Path

import pytest
import torch
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydanctic_if_types import PydanticFSDPModuleType, PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2Block
from modalities.running_env.cuda_env import CudaEnv

working_dir = Path(os.path.dirname(__file__))


class ActivationCheckpointingInstantiationModel(BaseModel):
    activation_checkpointed_model: PydanticFSDPModuleType
    wrapped_model: PydanticFSDPModuleType
    model_raw: PydanticPytorchModuleType


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_activation_checkpointing():
    config_file_path = working_dir / "config_activation_checkpointing.yaml"

    main = Main(config_file_path)
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
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
