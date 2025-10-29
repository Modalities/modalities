import json
import logging
import multiprocessing as py_mp
import os
import tempfile
import time
import traceback
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
from modalities.config.pydantic_if_types import PydanticAppStateType, PydanticPipelineType
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
    torch.cuda.device_count() < 4,
    reason="This e2e test requires 4 GPUs",
)
class TestFSDP2DCPCheckpointing:
    @staticmethod
    def _get_app_state(config_file_path: Path, use_pp: bool = False) -> AppState:
        if use_pp:

            class ComponentsInstantiationModel(BaseModel):
                app_state: PydanticAppStateType
                scheduled_pipeline: PydanticPipelineType

        else:

            class ComponentsInstantiationModel(BaseModel):
                app_state: PydanticAppStateType

        main_obj = Main(config_file_path)
        components: ComponentsInstantiationModel = main_obj.build_components(
            components_model_type=ComponentsInstantiationModel
        )
        app_state = components.app_state
        if use_pp:
            app_state.scheduled_pipeline = components.scheduled_pipeline
        return app_state

    @staticmethod
    @pytest.mark.parametrize(
        "config_filename,world_size,use_pp",
        [
            ("fsdp2_gpt2_config.yaml", 2, False),
            ("fsdp2_pp_gpt2_config.yaml", 2, True),
        ],
    )
    def test_save_checkpoint_after_backward_pass(
        temporary_checkpoint_folder_path: Path, config_filename: str, world_size: int, use_pp: bool
    ):
        working_dir = Path(os.path.dirname(__file__))
        config_file_path = working_dir / config_filename
        # Use a Manager queue so child processes can report exceptions to the parent.
        manager = py_mp.Manager()
        error_queue = manager.Queue()

        # Start child processes without joining so the parent can monitor a shared queue
        # and terminate remaining workers immediately if any child fails.
        proc_ctx = mp.spawn(
            TestFSDP2DCPCheckpointing._test_save_checkpoint_after_backward_pass_impl_wrapper,
            args=(world_size, temporary_checkpoint_folder_path, config_file_path, use_pp, error_queue),
            nprocs=world_size,
            join=False,
        )

        TestFSDP2DCPCheckpointing._monitor_child_processes(manager, error_queue, proc_ctx)

    @staticmethod
    def _test_save_checkpoint_after_backward_pass_impl_wrapper(
        process_id: int,
        world_size: int,
        temporary_checkpoint_folder_path: Path,
        gpt2_model_config_path: Path,
        use_pp: bool,
        error_queue: "py_mp.managers.SyncManager.Queue",
    ):
        # wraps the actual test function to be able to run it in a distributed  multiprocessing setup
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=22354,
        ):
            try:
                # build all the components for the test
                app_state1 = TestFSDP2DCPCheckpointing._get_app_state(gpt2_model_config_path, use_pp)
                app_state2 = TestFSDP2DCPCheckpointing._get_app_state(gpt2_model_config_path, use_pp)

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
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"Process {process_id} encountered an error:\n{e}")
                logging.error(tb)
                try:
                    error_queue.put((process_id, tb))
                except Exception:
                    logging.error("Failed to put exception info into error queue.")
                os._exit(1)

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
        if hasattr(app_state1, "scheduled_pipeline"):
            loss_0 = CheckpointingTestUtils.forward_backward_pp_pass(
                scheduled_pipeline=app_state1.scheduled_pipeline,
                optimizer=app_state1.optimizer,
                batch_input_ids_dict=batch_input_ids_dict,
                batch_target_ids=batch_target_ids,
            )
        else:
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
        if hasattr(app_state1, "scheduled_pipeline"):
            try:
                loss_1 = CheckpointingTestUtils.forward_backward_pp_pass(
                    scheduled_pipeline=app_state1.scheduled_pipeline,
                    optimizer=app_state1.optimizer,
                    batch_input_ids_dict=batch_input_ids_dict,
                    batch_target_ids=batch_target_ids,
                )
                loss_2 = CheckpointingTestUtils.forward_backward_pp_pass(
                    scheduled_pipeline=app_state2.scheduled_pipeline,
                    optimizer=app_state2.optimizer,
                    batch_input_ids_dict=batch_input_ids_dict,
                    batch_target_ids=batch_target_ids,
                )
            except Exception as e:
                print(f"Exception in _forward_step_with_pp: {e}")
                traceback.print_exc()
                raise
        else:
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
        if loss_1 is not None:
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

    @staticmethod
    def _monitor_child_processes(manager, error_queue, proc_ctx):
        # Normalize the return value from mp.spawn. When join=False it often
        # returns a ProcessContext-like object that may expose a `processes`
        # attribute. Other implementations may return an iterable of Process
        # objects. Build a `processes` list defensively so we can monitor and
        # terminate child processes below without assuming a particular type.
        processes = []
        if proc_ctx is None:
            processes = []
        else:
            # common attribute names that might hold the list of processes
            candidate_attrs = ["processes", "_processes", "workers", "process_list", "processes_"]
            found = False
            for attr in candidate_attrs:
                if hasattr(proc_ctx, attr):
                    ps = getattr(proc_ctx, attr)
                    try:
                        processes = list(ps)
                    except Exception:
                        processes = [ps]
                    found = True
                    break
            if not found:
                # If proc_ctx itself is iterable, exhaust it into a list
                try:
                    processes = list(proc_ctx)
                except Exception:
                    # Fallback: if proc_ctx behaves like a single process-like
                    # object (has terminate/is_alive/join), wrap it in a list.
                    if hasattr(proc_ctx, "terminate") or hasattr(proc_ctx, "is_alive") or hasattr(proc_ctx, "join"):
                        processes = [proc_ctx]
                    else:
                        processes = []

        # Monitor the error queue and child processes. If any child reports an exception,
        # terminate the other workers and raise the error in the parent to fail the test fast.
        try:
            # Loop until all processes finished or an error is reported
            while True:
                # If an error was reported by any child process, terminate remaining children
                if not error_queue.empty():
                    proc_id, tb = error_queue.get()
                    # terminate and join all processes (or the proc_ctx wrapper)
                    for p in processes:
                        try:
                            if hasattr(p, "is_alive"):
                                alive = p.is_alive()
                            elif hasattr(p, "exitcode"):
                                alive = getattr(p, "exitcode") is None
                            else:
                                alive = True
                            if alive and hasattr(p, "terminate"):
                                p.terminate()
                        except Exception:
                            pass
                    # If we didn't find individual process objects but proc_ctx
                    # exposes a terminate method, call it as a fallback.
                    try:
                        if not processes and hasattr(proc_ctx, "terminate"):
                            proc_ctx.terminate()
                    except Exception:
                        pass

                    for p in processes:
                        try:
                            if hasattr(p, "join"):
                                p.join(timeout=5)
                        except Exception:
                            pass
                    try:
                        if hasattr(proc_ctx, "join"):
                            proc_ctx.join(timeout=1)
                    except Exception:
                        pass
                    raise AssertionError(f"Child process {proc_id} raised an exception:\n{tb}")

                # If all processes have finished, break
                all_finished = all((not p.is_alive()) for p in processes)
                if all_finished:
                    # join them to collect exitcodes
                    for p in processes:
                        try:
                            p.join()
                        except Exception:
                            pass
                    # If we have a ProcessContext, call its join to clean up as well
                    try:
                        if hasattr(proc_ctx, "join"):
                            proc_ctx.join(timeout=1)
                    except Exception:
                        pass
                    break

                time.sleep(0.05)
        finally:
            try:
                manager.shutdown()
            except Exception:
                pass
