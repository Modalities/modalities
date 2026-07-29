from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
import torch
import torch.multiprocessing as mp
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from modalities.__main__ import load_app_config_dict
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, PydanticPytorchModuleType
from modalities.models.coca.coca_model import CoCa, CoCaConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.model_factory import ModelFactory
from modalities.optimizers import optimizer_factory as optimizer_factory_module
from modalities.optimizers.optimizer_factory import OptimizerFactory, get_optimizer_groups
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.env_utils import MixedPrecisionSettings
from tests.conftest import _ROOT_DIR
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv
from tests.utility import find_free_port

# number of parameters for each optimizer group
GPT2_LINEAR = 66130944
GPT2_EMBEDDING = 768 * (50304 + 2048)  # n_embd * (vocab_size + sequence_length)
GPT2_LAYERNORM = 768 * 50  # n_embd * num_layer_norms
COCA_ALL = 184502784


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires at least 1 CUDA device.")
@pytest.mark.parametrize(
    "model_name, weight_decay, weight_decay_groups_excluded, success,"
    "num_decayed_parameters, num_nondecayed_parameters",
    [
        ("gpt2", 0.0, [], True, 0, GPT2_LINEAR + GPT2_EMBEDDING + GPT2_LAYERNORM),
        ("gpt2", 1e-1, [], True, GPT2_LINEAR + GPT2_EMBEDDING + GPT2_LAYERNORM, 0),
        ("gpt2", 1e-1, ["embedding"], True, GPT2_LINEAR + GPT2_LAYERNORM, GPT2_EMBEDDING),
        ("gpt2", 1e-1, ["embedding", "layernorm"], True, GPT2_LINEAR, GPT2_EMBEDDING + GPT2_LAYERNORM),
        ("gpt2", 1e-1, ["linear", "embedding", "layernorm"], False, 0, GPT2_LINEAR + GPT2_EMBEDDING + GPT2_LAYERNORM),
        ("gpt2", 1e-1, ["non-existing-group"], False, None, None),
        ("coca", 0.0, [], True, 0, COCA_ALL),
        ("coca", 1e-1, [], True, COCA_ALL, 0),
        ("coca", 1e-1, ["non-existing-group"], False, None, None),
    ],
)
def test_get_optimizer_groups(
    model_name: Literal["gpt2"] | Literal["coca"],
    weight_decay: float,
    weight_decay_groups_excluded: list[str],
    success: bool,
    num_decayed_parameters: int,
    num_nondecayed_parameters: int,
):
    world_size = 1  # keep single-process semantics but use spawn for consistency
    port = find_free_port()
    mp.spawn(
        _run_single_optimizer_group_case,
        args=(
            world_size,
            port,
            model_name,
            weight_decay,
            weight_decay_groups_excluded,
            success,
            num_decayed_parameters,
            num_nondecayed_parameters,
        ),
        nprocs=world_size,
        join=True,
    )


def test_get_adam_builds_optimizer_from_groups(monkeypatch):
    wrapped_model = MagicMock()
    optimizer_groups = [{"params": [MagicMock()], "weight_decay": 0.1}]
    optimizer = MagicMock()
    get_optimizer_groups_mock = MagicMock(return_value=optimizer_groups)
    adam_mock = MagicMock(return_value=optimizer)

    monkeypatch.setattr(optimizer_factory_module, "get_optimizer_groups", get_optimizer_groups_mock)
    monkeypatch.setattr(optimizer_factory_module, "Adam", adam_mock)

    result = OptimizerFactory.get_adam(
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        weight_decay_groups_excluded=["embedding"],
        wrapped_model=wrapped_model,
        foreach=True,
        fused=False,
    )

    assert result is optimizer
    get_optimizer_groups_mock.assert_called_once_with(wrapped_model, 0.1, ["embedding"])
    adam_mock.assert_called_once_with(
        params=optimizer_groups,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=True,
        fused=False,
    )


def test_get_adam_w_builds_optimizer_from_groups(monkeypatch):
    wrapped_model = MagicMock()
    optimizer_groups = [{"params": [MagicMock()], "weight_decay": 0.2}]
    optimizer = MagicMock()
    get_optimizer_groups_mock = MagicMock(return_value=optimizer_groups)
    adam_w_mock = MagicMock(return_value=optimizer)

    monkeypatch.setattr(optimizer_factory_module, "get_optimizer_groups", get_optimizer_groups_mock)
    monkeypatch.setattr(optimizer_factory_module, "AdamW", adam_w_mock)

    result = OptimizerFactory.get_adam_w(
        lr=2e-4,
        betas=(0.8, 0.99),
        eps=1e-6,
        weight_decay=0.2,
        weight_decay_groups_excluded=["layernorm"],
        wrapped_model=wrapped_model,
        foreach=False,
        fused=True,
    )

    assert result is optimizer
    get_optimizer_groups_mock.assert_called_once_with(wrapped_model, 0.2, ["layernorm"])
    adam_w_mock.assert_called_once_with(
        params=optimizer_groups,
        lr=2e-4,
        betas=(0.8, 0.99),
        eps=1e-6,
        foreach=False,
        fused=True,
    )


def test_get_muon_builds_optimizer_from_groups(monkeypatch):
    wrapped_model = MagicMock()
    optimizer_groups = [{"params": [MagicMock()], "weight_decay": 0.05}]
    optimizer = MagicMock()
    get_optimizer_groups_mock = MagicMock(return_value=optimizer_groups)
    muon_mock = MagicMock(return_value=optimizer)

    monkeypatch.setattr(optimizer_factory_module, "get_optimizer_groups", get_optimizer_groups_mock)
    monkeypatch.setattr(optimizer_factory_module, "Muon", muon_mock)

    result = OptimizerFactory.get_muon(
        lr=3e-4,
        weight_decay=0.05,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(1.0, 0.5, 0.25),
        eps=1e-9,
        ns_steps=7,
        adjust_lr_fn="cosine",
        weight_decay_groups_excluded=["embedding"],
        wrapped_model=wrapped_model,
    )

    assert result is optimizer
    get_optimizer_groups_mock.assert_called_once_with(wrapped_model, 0.05, ["embedding"])
    muon_mock.assert_called_once_with(
        params=optimizer_groups,
        lr=3e-4,
        weight_decay=0.05,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(1.0, 0.5, 0.25),
        eps=1e-9,
        ns_steps=7,
        adjust_lr_fn="cosine",
    )


def test_get_fsdp1_checkpointed_optimizer_loads_optimizer_state():
    checkpoint_loading = MagicMock()
    checkpoint_path = Path("/tmp/checkpoint")
    wrapped_model = MagicMock()
    optimizer = MagicMock()

    result = OptimizerFactory.get_fsdp1_checkpointed_optimizer_(
        checkpoint_loading=checkpoint_loading,
        checkpoint_path=checkpoint_path,
        wrapped_model=wrapped_model,
        optimizer=optimizer,
    )

    assert result is optimizer
    checkpoint_loading.load_optimizer_checkpoint_.assert_called_once_with(
        file_path=checkpoint_path,
        optimizer=optimizer,
        model=wrapped_model,
    )


def _run_single_optimizer_group_case(
    process_id: int,
    world_size: int,
    port: int,
    model_name: str,
    weight_decay: float,
    weight_decay_groups_excluded: list[str],
    success: bool,
    num_decayed_parameters: int | None,
    num_nondecayed_parameters: int | None,
):
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=process_id,
        local_rank=process_id,
        world_size=world_size,
        rdvz_port=port,
    ):
        if model_name == "gpt2":
            model = _load_gpt2()
        elif model_name == "coca":
            model = _load_coca()
        else:
            raise ValueError(f"Unknown model_name={model_name}")

        if not success:
            with pytest.raises(Exception):
                get_optimizer_groups(model, weight_decay, weight_decay_groups_excluded)
            return

        optimizer_groups = get_optimizer_groups(model, weight_decay, weight_decay_groups_excluded)

        test_num_decayed_parameters = sum(
            p.numel() for group in optimizer_groups for p in group["params"] if group["weight_decay"] > 0
        )
        test_num_nondecayed_parameters = sum(
            p.numel() for group in optimizer_groups for p in group["params"] if group["weight_decay"] == 0
        )

        assert (
            test_num_decayed_parameters == num_decayed_parameters
        ), f"#(decayed parameters) = {test_num_decayed_parameters} should be {num_decayed_parameters}"
        assert (
            test_num_nondecayed_parameters == num_nondecayed_parameters
        ), f"#(non-decayed parameters) = {test_num_nondecayed_parameters} should be {num_nondecayed_parameters}"


def _load_gpt2() -> FSDP:
    config_file_path = _ROOT_DIR / Path("tests/test_yaml_configs/gpt2_config_optimizer.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    gpt2_model = _get_gpt2_model_from_config(config_dict)
    gpt2_wrapped_model = ModelFactory.get_fsdp1_wrapped_model(
        gpt2_model,
        sync_module_states=True,
        block_names=["GPT2Block"],
        mixed_precision_settings=MixedPrecisionSettings.FP_16,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    return gpt2_wrapped_model


def _get_gpt2_model_from_config(gpt2_model_config_dict: dict) -> GPT2LLM:
    class GPT2InstantationModel(BaseModel):
        model: PydanticPytorchModuleType

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    components = component_factory.build_components(
        config_dict=gpt2_model_config_dict, components_model_type=GPT2InstantationModel
    )

    model = components.model
    return model


def _load_coca() -> FSDP:
    config_file_path = _ROOT_DIR / Path("tests/models/coca/coca_config.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    coca_config = CoCaConfig.model_validate(config_dict)
    coca_model = CoCa(**dict(coca_config))
    coca_wrapped_model = ModelFactory.get_fsdp1_wrapped_model(
        coca_model,
        sync_module_states=True,
        block_names=["TransformerBlock", "VisionTransformerBlock"],
        mixed_precision_settings=MixedPrecisionSettings.FP_16,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    return coca_wrapped_model
