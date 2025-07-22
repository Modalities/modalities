import os
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2Block
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv

working_dir = Path(os.path.dirname(__file__))


class RawModel(BaseModel):
    model_raw: PydanticPytorchModuleType


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
def test_full_activation_checkpointing_FSDP1_legacy(world_size: int, rdvz_port: int, relative_config_path: str):
    # this test is for full activation checkpointing using the legacy FSDP1 implementation
    mp.spawn(
        _test_full_activation_checkpointing_FSDP1_legacy_thread,
        args=(rdvz_port, world_size, relative_config_path),
        nprocs=world_size,
        join=True,
    )


def _test_full_activation_checkpointing_FSDP1_legacy_thread(
    process_id: int, rdvz_port: int, world_size: int, relative_config_path: str
):
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
def test_full_activation_checkpointing_FSDPX(world_size: int, rdvz_port: int, relative_config_path: str):
    mp.spawn(
        _test_full_activation_checkpointing_FSDPX_thread,
        args=(rdvz_port, world_size, relative_config_path),
        nprocs=world_size,
        join=True,
    )


def _test_full_activation_checkpointing_FSDPX_thread(
    process_id: int, rdvz_port: int, world_size: int, relative_config_path: str
):
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
def test_fsdp2_full_activation_checkpointing(relative_config_path: str):
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
def test_fsdp2_selective_layer_activation_checkpointing(relative_config_path: str):
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
def test_fsdp2_selective_op_activation_checkpointing(relative_config_path: str):
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


# end to end equivalence test in terms of loss


@pytest.mark.parametrize(
    "relative_config_path",
    [
        ("config_activation_checkpointing.yaml"),
    ],
)
def test_fsdp2_activation_checkpointing_end2end(relative_config_path: str):
    def forward_and_backward(model: nn.Module, input_ids: torch.Tensor) -> float:
        target = input_ids[:, 1:]  # batch_size, seq_len - 1
        input_ids = input_ids[:, :-1]  # batch_size, seq_len - 1
        input_dict = {"input_ids": input_ids}
        logits = model(input_dict)["logits"]  # batch_size, seq_len - 1, vocab_size

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # batch_size * (seq_len - 1), vocab_size
            target.reshape(-1),  # batch_size * (seq_len - 1)
            reduction="mean",
        )
        loss_val = loss.item()
        loss.backward()
        return loss_val

    def check_grads_equal(model1, model2, label):
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                # we cannot check the FQNs as AC renames the parameters.
                # inestead we check for weight equivalence
                torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-7, msg=f"Parameter mismatch in {n1} ({label})")
                torch.testing.assert_close(
                    p1.grad, p2.grad, rtol=1e-5, atol=1e-7, msg=f"Gradient mismatch in {n1} ({label})"
                )

    batch_size = 2
    seq_len = 256
    vocab_size = 50304

    # build the models with different activation checkpointing variants but equivalent weights
    config_file_path = working_dir / relative_config_path
    main = Main(config_file_path, experiment_id="-1")

    torch.manual_seed(42)
    model_raw = main.build_components(components_model_type=RawModel).model_raw.to("cuda")

    torch.manual_seed(42)
    model_fac = main.build_components(
        components_model_type=FullActivationCheckpointingInstantiationModel
    ).full_activation_checkpointed_model.to("cuda")

    torch.manual_seed(42)
    model_sel_layer = main.build_components(
        components_model_type=SelectiveLayerActivationCheckpointingInstantiationModel
    ).selective_layer_activation_checkpointed_model.to("cuda")

    torch.manual_seed(42)
    model_sel_op = main.build_components(
        components_model_type=SelectiveOpActivationCheckpointingInstantiationModel
    ).selective_op_activation_checkpointed_model.to("cuda")

    # Ensure all models have a different reference
    models = [model_raw, model_fac, model_sel_layer, model_sel_op]
    assert len(set(id(m) for m in models)) == len(models)

    # Dummy LLM token input
    # we use a sequence length of seq_len + 1 as the last token will be only used for loss calculation
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device="cuda")

    # Run forward+backward
    loss_raw = forward_and_backward(model_raw, input_ids)
    loss_fac = forward_and_backward(model_fac, input_ids)
    loss_sel_layer = forward_and_backward(model_sel_layer, input_ids)
    loss_sel_op = forward_and_backward(model_sel_op, input_ids)

    # Compare losses
    torch.testing.assert_close(torch.tensor(loss_fac), torch.tensor(loss_raw), msg="FAC loss mismatch")
    torch.testing.assert_close(torch.tensor(loss_sel_layer), torch.tensor(loss_raw), msg="Sel layer AC loss mismatch")
    torch.testing.assert_close(torch.tensor(loss_sel_op), torch.tensor(loss_raw), msg="Sel op AC loss mismatch")

    # Compare gradients
    check_grads_equal(model_raw, model_fac, "fac")
    check_grads_equal(model_raw, model_sel_layer, "sel_layer")
    check_grads_equal(model_raw, model_sel_op, "sel_op")
