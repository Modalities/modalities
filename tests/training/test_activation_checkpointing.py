import os
from pathlib import Path

import pytest
import torch.multiprocessing as mp
from pydantic import BaseModel
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2Block
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv

working_dir = Path(os.path.dirname(__file__))


class ActivationCheckpointingInstantiationModel(BaseModel):
    test_model: PydanticPytorchModuleType


class FullActivationCheckpointingInstantiationModel(BaseModel):
    full_activation_checkpointed_model: PydanticPytorchModuleType


class SelectiveLayerActivationCheckpointingInstantiationModel(BaseModel):
    selective_layer_activation_checkpointed_model: PydanticPytorchModuleType


class SelectiveOpActivationCheckpointingInstantiationModel(BaseModel):
    selective_op_activation_checkpointed_model: PydanticPytorchModuleType


@pytest.mark.parametrize(
    "rdvz_port, world_size, relative_config_path",
    [
        (22310, 2, "config_activation_checkpointing_fsdp1_legacy.yaml"),
    ],
)
def test_selective_activation_checkpointing_FSDP1_legacy(world_size: int, rdvz_port: int, relative_config_path: str):
    mp.spawn(
        _test_selective_activation_checkpointing_FSDP1_legacy_thread,
        args=(rdvz_port, world_size, relative_config_path),
        nprocs=world_size,
        join=True,
    )


def _test_selective_activation_checkpointing_FSDP1_legacy_thread(
    process_id: int, rdvz_port: int, world_size: int, relative_config_path: str
):
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
        components: ActivationCheckpointingInstantiationModel = main.build_components(
            components_model_type=ActivationCheckpointingInstantiationModel
        )
        modules = dict(components.test_model.named_modules())

        # make sure that we have two modules that are wrapped with activation checkpointing
        # i.e., the two GPT2Block modules defined in the config
        ac_modules = [(module_name, module) for module_name, module in modules.items() if isinstance(module, GPT2Block)]
        assert len(ac_modules) == 2
        assert all([module_name.endswith("_checkpoint_wrapped_module") for module_name, _ in ac_modules])
        # make sure that there are no other modules that are wrapped with activation checkpointing
        assert sum([module_name.endswith("_checkpoint_wrapped_module") for module_name in modules.keys()]) == len(
            ac_modules
        )


@pytest.mark.parametrize(
    "rdvz_port, world_size, relative_config_path",
    [
        (22311, 2, "config_activation_checkpointing_fsdp1.yaml"),
        (22312, 2, "config_activation_checkpointing_fsdp2.yaml"),
    ],
)
def test_selective_activation_checkpointing_FSDPX(world_size: int, rdvz_port: int, relative_config_path: str):
    mp.spawn(
        _test_selective_activation_checkpointing_FSDPX_thread,
        args=(rdvz_port, world_size, relative_config_path),
        nprocs=world_size,
        join=True,
    )


def _test_selective_activation_checkpointing_FSDPX_thread(
    process_id: int, rdvz_port: int, world_size: int, relative_config_path: str
):
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
        components: ActivationCheckpointingInstantiationModel = main.build_components(
            components_model_type=ActivationCheckpointingInstantiationModel
        )
        modules = components.test_model.named_modules()
        for module_name, module in modules:
            if module_name in [
                "_fsdp_wrapped_module.transformer.h.0",
                "_fsdp_wrapped_module.transformer.h.1",  # FSDP1
            ] or module_name in [
                "transformer.h.0",
                "transformer.h.1",  # FSDP2
            ]:
                assert isinstance(module, CheckpointWrapper)
            else:
                assert not isinstance(module, CheckpointWrapper)


@pytest.mark.parametrize(
    "relative_config_path",
    [
        ("config_activation_checkpointing.yaml"),
    ],
)
def test_full_activation_checkpointing(relative_config_path: str):
    working_dir = Path(os.path.dirname(__file__))
    config_file_path = working_dir / relative_config_path

    main = Main(config_file_path, experiment_id="-1")
    components: FullActivationCheckpointingInstantiationModel = main.build_components(
        components_model_type=FullActivationCheckpointingInstantiationModel
    )
    modules = components.full_activation_checkpointed_model.named_modules()
    for module_name, module in modules:
        if module_name in ["transformer.h.0", "transformer.h.1", "transformer.h.2", "transformer.h.3"]:
            assert isinstance(module, CheckpointWrapper)
        else:
            assert not isinstance(module, CheckpointWrapper)


@pytest.mark.parametrize(
    "relative_config_path",
    [
        ("config_activation_checkpointing.yaml"),
    ],
)
def test_selective_layer_activation_checkpointing(relative_config_path: str):
    working_dir = Path(os.path.dirname(__file__))
    config_file_path = working_dir / relative_config_path

    main = Main(config_file_path, experiment_id="-1")
    components: SelectiveLayerActivationCheckpointingInstantiationModel = main.build_components(
        components_model_type=SelectiveLayerActivationCheckpointingInstantiationModel
    )
    modules = components.selective_layer_activation_checkpointed_model.named_modules()
    for module_name, module in modules:
        if module_name in ["transformer.h.1", "transformer.h.3"]:
            assert isinstance(module, CheckpointWrapper)
        else:
            assert not isinstance(module, CheckpointWrapper)


@pytest.mark.parametrize(
    "relative_config_path",
    [
        ("config_activation_checkpointing.yaml"),
    ],
)
def test_selective_op_activation_checkpointing(relative_config_path: str):
    working_dir = Path(os.path.dirname(__file__))
    config_file_path = working_dir / relative_config_path

    main = Main(config_file_path, experiment_id="-1")
    components: SelectiveOpActivationCheckpointingInstantiationModel = main.build_components(
        components_model_type=SelectiveOpActivationCheckpointingInstantiationModel
    )
    modules = components.selective_op_activation_checkpointed_model.named_modules()
    for module_name, module in modules:
        if module_name in ["transformer.h.0", "transformer.h.1", "transformer.h.2", "transformer.h.3"]:
            # TODO: we should add some checks here that check on an op-level if the checkpointing is applied
            assert isinstance(module, CheckpointWrapper)
        else:
            assert not isinstance(module, CheckpointWrapper)
