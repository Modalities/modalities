import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.checkpointing.checkpoint_loading import DistributedCheckpointLoadingIF
from modalities.checkpointing.checkpoint_saving_execution import CheckpointSavingExecutionABC
from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import DCPCheckpointLoading
from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import DCPCheckpointSaving
from modalities.checkpointing.stateful.app_state import AppState
from modalities.config.config import ProcessGroupBackendType, load_app_config_dict
from modalities.config.pydantic_if_types import PydanticAppStateType
from modalities.training.training_progress import TrainingProgress
from tests.checkpointing.checkpointing_test_utils import CheckpointingTestUtils
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@pytest.fixture
def temporary_checkpoint_folder_path():
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


@pytest.fixture
def gpt2_model_config_path() -> Path:
    working_dir = Path(os.path.dirname(__file__))
    config_file_path = working_dir / "fsdp2_gpt2_config.yaml"
    return config_file_path


def get_gpt2_model_config_dict(gpt2_model_config_path: Path) -> dict:
    config_dict = load_app_config_dict(config_file_path=gpt2_model_config_path)
    return config_dict


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs",
)
class TestFSDP2DCPCheckpointing:
    @staticmethod
    def _get_app_state(config_file_path: Path) -> AppState:
        class ComponentsInstantiationModel(BaseModel):
            app_state: PydanticAppStateType

        main_obj = Main(config_file_path)
        components: ComponentsInstantiationModel = main_obj.build_components(
            components_model_type=ComponentsInstantiationModel
        )
        return components.app_state

    @staticmethod
    def test_save_checkpoint_after_backward_pass(temporary_checkpoint_folder_path: Path, gpt2_model_config_path: Path):
        world_size = 2
        mp.spawn(
            TestFSDP2DCPCheckpointing._test_save_checkpoint_after_backward_pass_impl_wrapper,
            args=(world_size, temporary_checkpoint_folder_path, gpt2_model_config_path),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def _test_save_checkpoint_after_backward_pass_impl_wrapper(
        process_id: int,
        world_size: int,
        temporary_checkpoint_folder_path: Path,
        gpt2_model_config_path: Path,
    ):
        # wraps the actual test function to be able to run it in a distributed  multiprocessing setup
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=22356,
        ):
            # build all the components for the test
            app_state1 = TestFSDP2DCPCheckpointing._get_app_state(config_file_path=gpt2_model_config_path)
            app_state2 = TestFSDP2DCPCheckpointing._get_app_state(config_file_path=gpt2_model_config_path)

            gpt2_model_config_dict = get_gpt2_model_config_dict(gpt2_model_config_path=gpt2_model_config_path)
            experiment_id = "0"
            checkpoint_loading = DCPCheckpointLoading(global_rank=process_id)
            checkpoint_saving = DCPCheckpointSaving(
                checkpoint_path=temporary_checkpoint_folder_path,
                experiment_id=experiment_id,
                global_rank=process_id,
            )

            # run the test
            TestFSDP2DCPCheckpointing._test_save_checkpoint_after_backward_pass_impl(
                app_state1=app_state1,
                app_state2=app_state2,
                gpt2_model_config_dict=gpt2_model_config_dict,
                checkpoint_loading=checkpoint_loading,
                checkpoint_saving=checkpoint_saving,
                temporary_checkpoint_folder_path=temporary_checkpoint_folder_path,
                experiment_id=experiment_id,
            )

    @staticmethod
    def _test_save_checkpoint_after_backward_pass_impl(
        app_state1: AppState,
        app_state2: AppState,
        gpt2_model_config_dict: dict,
        checkpoint_loading: DistributedCheckpointLoadingIF,
        checkpoint_saving: CheckpointSavingExecutionABC,
        temporary_checkpoint_folder_path: Path,
        experiment_id: str,
    ):
        # Test setup:
        # 1. Create two app states with the same model and optimizer (difference references)
        # 2. Run a forward and backward pass on the first app state
        # 3. Save the model and optimizer state dicts as a checkpoint on disc
        # 4. Load the checkpoint into the second app state
        # 5. Run a forward and backward pass on both app states

        # Assertions:
        # 1. Check that the initial model is different from the updated ones
        # 2. Check that the model and optimizer state dicts after loading the checkpoint are the same as
        #    the updated ones
        # 3. Check that the model and optimizer state dicts after loading the checkpoint are the same as
        #    the updated ones after another forward pass
        # 4. Check that the model and optimiizer state dicts from the 2nd forwward and backward pass
        #    are different from the 1st forward and backward pass

        prediction_key = gpt2_model_config_dict["model_raw"]["config"]["prediction_key"]

        # save the initial model and optimizer state dicts
        untrained_model_parameters = CheckpointingTestUtils.clone_parameters(app_state1.model)
        untrained_optimizer_state_dict = deepcopy(app_state1.optimizer.state_dict())

        # run backward pass
        batch_input_ids_dict, batch_target_ids = CheckpointingTestUtils.generate_batch(gpt2_model_config_dict)
        loss_0 = CheckpointingTestUtils.forward_backward_pass(
            prediction_key=prediction_key,
            model=app_state1.model,
            optimizer=app_state1.optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )

        # save the updated model and optimizer states for later comparisons
        updated_model_parameters = CheckpointingTestUtils.clone_parameters(app_state1.model)
        updated_optimizer_state_dict = deepcopy(app_state1.optimizer.state_dict())

        # checkpoint the model and optimizer before backward pass
        local_micro_batch_size = batch_input_ids_dict["input_ids"].shape[0]
        sequence_length = batch_input_ids_dict["input_ids"].shape[1]
        num_ranks = torch.distributed.get_world_size()
        gradient_accumulation_steps = 1
        num_train_steps_done = 1
        num_seen_tokens_current_run = (
            num_train_steps_done * local_micro_batch_size * sequence_length * num_ranks * gradient_accumulation_steps
        )
        training_progress = TrainingProgress(
            num_seen_steps_current_run=num_train_steps_done,
            num_seen_tokens_current_run=num_seen_tokens_current_run,
            num_target_steps=num_train_steps_done + 1,
            num_target_tokens=num_seen_tokens_current_run * 2,
        )
        checkpoint_saving._save_checkpoint(app_state=app_state1, training_progress=training_progress)

        # check that the checkpoint was saved on each rank
        dcp_checkpoint_folder_path = (
            temporary_checkpoint_folder_path
            / experiment_id
            / "eid_0-seen_steps_1-seen_tokens_4096-target_steps_2-target_tokens_8192"
        )
        dcp_checkpoint_file_paths = list(dcp_checkpoint_folder_path.glob("*.distcp"))
        assert (
            len(dcp_checkpoint_file_paths) == torch.distributed.get_world_size()
        ), "There must be one checkpoint per rank."

        last_checkpoint_info_path = temporary_checkpoint_folder_path / experiment_id / "last_checkpoint_info.json"
        assert last_checkpoint_info_path.exists(), "last_checkpoint_info.json file must exist."
        # load checkpoint info
        with open(last_checkpoint_info_path, "r") as f:
            last_checkpoint_info = json.load(f)
        assert (
            Path(last_checkpoint_info["checkpoint_folder_path"]) == dcp_checkpoint_folder_path
        ), "checkpoint folder path not set correctly in checkpoint info file"

        # load the checkpoint
        checkpoint_loading.load_checkpoint_(
            app_state=app_state2,
            checkpoint_dir_path=dcp_checkpoint_folder_path,
        )

        loaded_and_updated_model_parameters = CheckpointingTestUtils.clone_parameters(app_state1.model)
        loaded_and_updated_optimizer_state_dict = deepcopy(app_state1.optimizer.state_dict())

        # perform another forward pass and backward pass for the previous and the loaded model
        loss_1 = CheckpointingTestUtils.forward_backward_pass(
            prediction_key=prediction_key,
            model=app_state1.model,
            optimizer=app_state1.optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )

        loss_2 = CheckpointingTestUtils.forward_backward_pass(
            prediction_key=prediction_key,
            model=app_state2.model,
            optimizer=app_state2.optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )
        assert loss_1 == loss_2, f"loss_1 = {loss_1} does not equal loss_2 = {loss_2}"
        assert loss_1 < loss_0, f"loss_1 = {loss_1} is not less than loss_0 = {loss_0}"

        # check that the model and optimizer states after each backward pass are as expected
        # model weights
        CheckpointingTestUtils.assert_equality_two_models(
            untrained_model_parameters, updated_model_parameters, must_be_equal=False
        )
        CheckpointingTestUtils.assert_equality_two_models(
            loaded_and_updated_model_parameters, updated_model_parameters, must_be_equal=True
        )
        CheckpointingTestUtils.assert_equality_two_models(
            app_state1.model.parameters(), app_state2.model.parameters(), must_be_equal=True
        )
        CheckpointingTestUtils.assert_equality_two_models(
            app_state1.model.parameters(), updated_model_parameters, must_be_equal=False
        )

        # param groups
        CheckpointingTestUtils.assert_equality_optimizer_param_group(
            untrained_optimizer_state_dict, updated_optimizer_state_dict, must_be_equal=True
        )
        CheckpointingTestUtils.assert_equality_optimizer_param_group(
            loaded_and_updated_optimizer_state_dict, updated_optimizer_state_dict, must_be_equal=True
        )
        CheckpointingTestUtils.assert_equality_optimizer_param_group(
            app_state1.optimizer.state_dict(), app_state2.optimizer.state_dict(), must_be_equal=True
        )
        # optimizer state
        CheckpointingTestUtils.assert_equality_optimizer_state(
            loaded_and_updated_optimizer_state_dict, updated_optimizer_state_dict, must_be_equal=True
        )
        CheckpointingTestUtils.assert_equality_optimizer_state(
            app_state1.optimizer.state_dict(), app_state2.optimizer.state_dict(), must_be_equal=True
        )
        CheckpointingTestUtils.assert_equality_optimizer_state(
            app_state1.optimizer.state_dict(), updated_optimizer_state_dict, must_be_equal=False
        )
