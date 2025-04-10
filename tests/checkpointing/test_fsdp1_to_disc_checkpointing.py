import os
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import AdamW, Optimizer

from modalities.__main__ import load_app_config_dict
from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import FSDP1CheckpointLoading
from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import CheckpointingEntityType, FSDP1CheckpointSaving
from modalities.checkpointing.stateful.app_state import AppState
from modalities.config.config import ProcessGroupBackendType
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2LLMConfig
from modalities.models.model_factory import ModelFactory
from modalities.optimizers.optimizer_factory import OptimizerFactory
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.training.training_progress import TrainingProgress
from tests.checkpointing.checkpointing_test_utils import CheckpointingTestUtils
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv

working_dir = Path(os.path.dirname(__file__))


def get_gpt2_model(gpt2_model_config_dict: GPT2LLMConfig) -> GPT2LLM:
    return CheckpointingTestUtils.get_gpt2_model_from_config(gpt2_model_config_dict)


def get_fsdp1_wrapped_model(gpt2_model: GPT2LLM) -> FSDP:
    wrapped_model: FSDP = ModelFactory.get_fsdp1_wrapped_model(
        gpt2_model,
        sync_module_states=True,
        block_names=["GPT2Block"],
        mixed_precision_settings=MixedPrecisionSettings.FP_16,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )
    return wrapped_model


def get_optimizer(fsdp1_wrapped_model: nn.Module) -> Optimizer:
    optimizer = OptimizerFactory.get_adam_w(
        wrapped_model=fsdp1_wrapped_model,
        lr=0.001,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=1e-1,
        weight_decay_groups_excluded=[],
    )
    return optimizer


@pytest.fixture
def temporary_checkpoint_folder_path():
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


@pytest.fixture
def gpt2_model_config_dict() -> dict:
    config_file_path = working_dir / "gpt2_config.yaml"
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    return config_dict


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
class TestFSDP1ToDiscCheckpointing:
    @staticmethod
    def _clone_parameters(fsdp_wrapped_model: FSDP):
        return [p.clone() for p in fsdp_wrapped_model.parameters() if p.requires_grad and p.numel() > 0]

    @staticmethod
    def _assert_equality_optimizer_param_group(
        optimizer_1_state_dict: dict, optimizer_2_state_dict: dict, must_be_equal: bool
    ):
        if must_be_equal:
            assert (
                optimizer_1_state_dict["param_groups"] == optimizer_2_state_dict["param_groups"]
            ), "_assert_equality_optimizer_param_group failed (must_be_equal = True)"
        else:
            assert not (
                optimizer_1_state_dict["param_groups"] == optimizer_2_state_dict["param_groups"]
            ), "_assert_equality_optimizer_param_group failed (must_be_equal = False)"

    @staticmethod
    def _assert_equality_optimizer_state(
        optimizer_1_state_dict: dict, optimizer_2_state_dict: dict, must_be_equal: bool
    ):
        optimizer_1_state = optimizer_1_state_dict["state"]
        optimizer_2_state = optimizer_2_state_dict["state"]
        assert set(optimizer_1_state.keys()) == set(optimizer_2_state.keys())

        for param_group_id in optimizer_1_state.keys():
            state_1 = optimizer_1_state[param_group_id]
            state_2 = optimizer_2_state[param_group_id]
            assert set(state_1.keys()) == set(state_2.keys())
            for state_key in state_1.keys():
                if must_be_equal:
                    assert torch.equal(
                        state_1[state_key], state_2[state_key]
                    ), "_assert_equality_optimizer_state failed (must_be_equal = True)"
                else:
                    assert not torch.equal(
                        state_1[state_key], state_2[state_key]
                    ), "_assert_equality_optimizer_state failed (must_be_equal = False)"

    @staticmethod
    def _assert_equality_two_models(params_1: list[torch.Tensor], params_2: list[torch.Tensor], must_be_equal: bool):
        for p1, p2 in zip(params_1, params_2):
            if must_be_equal:
                assert torch.equal(p1, p2), "_assert_equality_two_models failed (must_be_equal = True)"
            else:
                assert not torch.equal(p1, p2), "_assert_equality_two_models failed (must_be_equal = False)"

    @staticmethod
    def test_save_checkpoint_after_backward_pass(temporary_checkpoint_folder_path: Path, gpt2_model_config_dict: dict):
        world_size = 2
        mp.spawn(
            TestFSDP1ToDiscCheckpointing._test_save_checkpoint_after_backward_pass_impl_wrapper,
            args=(world_size, temporary_checkpoint_folder_path, gpt2_model_config_dict),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def _test_save_checkpoint_after_backward_pass_impl_wrapper(
        process_id: int, world_size: int, temporary_checkpoint_folder_path: Path, gpt2_model_config_dict: dict
    ):
        # wraps the actual test function to be able to run it in a distributed  multiprocessing setup
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=22355,
        ):
            gpt2_model = get_gpt2_model(gpt2_model_config_dict=gpt2_model_config_dict)
            fsdp1_wrapped_model = get_fsdp1_wrapped_model(gpt2_model=gpt2_model)
            optimizer = get_optimizer(fsdp1_wrapped_model=fsdp1_wrapped_model)

            gpt2_model_2 = get_gpt2_model(gpt2_model_config_dict=gpt2_model_config_dict)

            TestFSDP1ToDiscCheckpointing._test_save_checkpoint_after_backward_pass_impl(
                fsdp1_wrapped_model=fsdp1_wrapped_model,
                optimizer=optimizer,
                temporary_checkpoint_folder_path=temporary_checkpoint_folder_path,
                gpt2_model_2=gpt2_model_2,
                gpt2_model_config_dict=gpt2_model_config_dict,
            )

    @staticmethod
    def _test_save_checkpoint_after_backward_pass_impl(
        fsdp1_wrapped_model: FSDP,
        optimizer: Optimizer,
        temporary_checkpoint_folder_path: Path,
        gpt2_model_2: GPT2LLM,
        gpt2_model_config_dict: dict,
    ):
        experiment_id = "0"
        num_train_steps_done = 1
        num_ranks = 2
        local_micro_batch_size = 4
        gradient_accumulation_steps = 1
        sequence_length = gpt2_model_config_dict["model_raw"]["config"]["sequence_length"]

        checkpoint_saving = FSDP1CheckpointSaving(
            checkpoint_path=temporary_checkpoint_folder_path,
            experiment_id=experiment_id,
            global_rank=dist.get_rank(),
        )

        checkpoint_loading = FSDP1CheckpointLoading(
            global_rank=dist.get_rank(),
            block_names=["GPT2Block"],
            mixed_precision_settings=MixedPrecisionSettings.FP_16,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )

        untrained_model_parameters = TestFSDP1ToDiscCheckpointing._clone_parameters(fsdp1_wrapped_model)
        untrained_optimizer_state_dict = deepcopy(optimizer.state_dict())

        # run backward pass
        batch_input_ids_dict, batch_target_ids = CheckpointingTestUtils.generate_batch(gpt2_model_config_dict)
        CheckpointingTestUtils.forward_backward_pass(
            gpt2_model_config=gpt2_model_config_dict,
            model=fsdp1_wrapped_model,
            optimizer=optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )
        updated_model_parameters = TestFSDP1ToDiscCheckpointing._clone_parameters(fsdp1_wrapped_model)
        updated_optimizer_state_dict = deepcopy(optimizer.state_dict())

        # save model and optimizer before backward pass
        training_progress = TrainingProgress(
            num_seen_steps_current_run=num_train_steps_done,
            num_seen_tokens_current_run=num_train_steps_done
            * local_micro_batch_size
            * sequence_length
            * num_ranks
            * gradient_accumulation_steps,
            num_target_steps=num_train_steps_done * 2,
            num_target_tokens=num_train_steps_done
            * local_micro_batch_size
            * sequence_length
            * num_ranks
            * gradient_accumulation_steps
            * 2,
        )
        app_state = AppState(model=fsdp1_wrapped_model, optimizer=optimizer)
        checkpoint_saving._save_checkpoint(app_state=app_state, training_progress=training_progress)

        # load the model checkpoint
        model_checkpointing_path = checkpoint_saving._get_checkpointing_path(
            experiment_id=experiment_id,
            entity_type=CheckpointingEntityType.MODEL,
            num_seen_steps=training_progress.num_seen_steps_total,
            num_seen_tokens=training_progress.num_seen_tokens_total,
            num_target_steps=training_progress.num_target_steps,
            num_target_tokens=training_progress.num_target_tokens,
        )
        fsdp_wrapped_model_2 = checkpoint_loading.load_model_checkpoint(
            model=gpt2_model_2, file_path=model_checkpointing_path
        )

        optimizer_2 = AdamW(fsdp_wrapped_model_2.parameters(), lr=0.001)

        optimizer_checkpointing_path = checkpoint_saving._get_checkpointing_path(
            experiment_id=experiment_id,
            entity_type=CheckpointingEntityType.OPTIMIZER,
            num_seen_steps=training_progress.num_seen_steps_total,
            num_seen_tokens=training_progress.num_seen_tokens_total,
            num_target_steps=training_progress.num_target_steps,
            num_target_tokens=training_progress.num_target_tokens,
        )
        checkpoint_loading.load_optimizer_checkpoint_(
            optimizer=optimizer_2, model=fsdp_wrapped_model_2, file_path=optimizer_checkpointing_path
        )

        loaded_and_updated_model_parameters = TestFSDP1ToDiscCheckpointing._clone_parameters(fsdp1_wrapped_model)
        loaded_and_updated_optimizer_state_dict = deepcopy(optimizer_2.state_dict())

        # make sure that after the update all weights are DIFFERENT from the original ones
        TestFSDP1ToDiscCheckpointing._assert_equality_two_models(
            updated_model_parameters, untrained_model_parameters, must_be_equal=False
        )
        TestFSDP1ToDiscCheckpointing._assert_equality_optimizer_param_group(
            updated_optimizer_state_dict, untrained_optimizer_state_dict, must_be_equal=True
        )

        # make sure that the updated parameters are EQUAL to the ones that we saved subsequently
        TestFSDP1ToDiscCheckpointing._assert_equality_two_models(
            updated_model_parameters, loaded_and_updated_model_parameters, must_be_equal=True
        )
        TestFSDP1ToDiscCheckpointing._assert_equality_optimizer_param_group(
            updated_optimizer_state_dict, loaded_and_updated_optimizer_state_dict, must_be_equal=True
        )
        TestFSDP1ToDiscCheckpointing._assert_equality_optimizer_state(
            updated_optimizer_state_dict, loaded_and_updated_optimizer_state_dict, must_be_equal=True
        )

        # we do another forward/backward pass and check
        #  if the weights are equally updated for the loaded model as for the not-loaded model
        # run backward pass
        batch_input_ids_dict, batch_target_ids = CheckpointingTestUtils.generate_batch(gpt2_model_config_dict)

        loss_1 = CheckpointingTestUtils.forward_backward_pass(
            gpt2_model_config=gpt2_model_config_dict,
            model=fsdp1_wrapped_model,
            optimizer=optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )
        loss_2 = CheckpointingTestUtils.forward_backward_pass(
            gpt2_model_config=gpt2_model_config_dict,
            model=fsdp_wrapped_model_2,
            optimizer=optimizer_2,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )

        assert loss_1 == loss_2, f"loss_1 = {loss_1} does not equal loss_2 = {loss_2}"

        # make sure that after another update the two models and optimizers are the same
        TestFSDP1ToDiscCheckpointing._assert_equality_two_models(
            fsdp1_wrapped_model.parameters(), fsdp_wrapped_model_2.parameters(), must_be_equal=True
        )
        TestFSDP1ToDiscCheckpointing._assert_equality_optimizer_param_group(
            optimizer.state_dict(), optimizer_2.state_dict(), must_be_equal=True
        )
        TestFSDP1ToDiscCheckpointing._assert_equality_optimizer_state(
            optimizer.state_dict(), optimizer_2.state_dict(), must_be_equal=True
        )

        # make sure that the weights and state has changed to the previous forward backward pass
        TestFSDP1ToDiscCheckpointing._assert_equality_two_models(
            fsdp1_wrapped_model.parameters(), updated_model_parameters, must_be_equal=False
        )
        TestFSDP1ToDiscCheckpointing._assert_equality_optimizer_param_group(
            optimizer.state_dict(), updated_optimizer_state_dict, must_be_equal=True
        )
        TestFSDP1ToDiscCheckpointing._assert_equality_optimizer_state(
            optimizer.state_dict(), updated_optimizer_state_dict, must_be_equal=False
        )
