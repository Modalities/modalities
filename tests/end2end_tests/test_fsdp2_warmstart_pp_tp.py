import json
import logging
import multiprocessing as py_mp
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import BaseModel

from modalities.__main__ import Main, load_app_config_dict
from modalities.batch import EvaluationResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.config.pydantic_if_types import PydanticLLMDataLoaderIFType
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import Message
from tests.end2end_tests.custom_components import (
    MultiProcessingCudaEnv,
    SaveAllResultSubscriber,
    SaveAllResultSubscriberConfig,
)

working_dir = Path(os.path.dirname(__file__))
tmp_folder = working_dir / "../tmp/fsdp2_warmstart_pp_tp"
working_dir = working_dir / "configs"


class TrainDataloaderInstantiationModel(BaseModel):
    settings: TrainingComponentsInstantiationModel.Settings
    train_dataloader: PydanticLLMDataLoaderIFType


@pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="This e2e test requires 8 GPUs.",
)
class TestWarmstart:

    @pytest.mark.parametrize(
        "first_config,second_config,world_size_first,world_size_second",
        [
            ("gpt2_train_num_steps_7_pp_tp.yaml", "gpt2_warm_start_from_step_4_pp_tp.yaml", 8, 8),
            ("gpt2_train_num_steps_7_pp_tp.yaml", "gpt2_warm_start_from_step_4_fsdp2.yaml", 8, 2),
            ("gpt2_train_num_steps_7_pp_tp.yaml", "gpt2_warm_start_from_step_4_grad_accu.yaml", 8, 1),
            ("gpt2_train_num_steps_7_grad_accu.yaml", "gpt2_warm_start_from_step_4_pp_tp.yaml", 1, 8),
        ],
    )
    def test_warm_start(self, first_config: str, second_config: str, world_size_first: int, world_size_second: int):
        # Sequential two-phase training test using multiprocessing.
        try:
            if tmp_folder.exists():
                shutil.rmtree(tmp_folder)
            tmp_folder.mkdir(parents=False, exist_ok=False)

            # ---- First training phase ----
            manager_first = py_mp.Manager()
            error_queue_first = manager_first.Queue()
            proc_ctx_first = mp.spawn(
                TestWarmstart._first_training_impl_wrapper,
                args=(world_size_first, first_config, tmp_folder, error_queue_first),
                nprocs=world_size_first,
                join=False,
            )
            TestWarmstart._monitor_child_processes(manager_first, error_queue_first, proc_ctx_first)

            # ---- Second (warmstart) training phase ----
            manager_second = py_mp.Manager()
            error_queue_second = manager_second.Queue()
            proc_ctx_second = mp.spawn(
                TestWarmstart._second_training_impl_wrapper,
                args=(world_size_second, second_config, tmp_folder, error_queue_second),
                nprocs=world_size_second,
                join=False,
            )
            TestWarmstart._monitor_child_processes(manager_second, error_queue_second, proc_ctx_second)
        finally:
            try:
                if tmp_folder.exists():
                    shutil.rmtree(tmp_folder)
            except Exception as e:
                logging.warning(f"Failed to remove tmp folder {tmp_folder}: {e}")

    # ---------- First training (from scratch) wrappers ----------

    @staticmethod
    def _first_training_impl_wrapper(
        process_id: int,
        world_size_first: int,
        first_config: str,
        checkpoint_root_path: Path,
        error_queue: Any,
    ):
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size_first,
            rdvz_port=24571,
        ):
            try:
                TestWarmstart._first_training_impl(
                    process_id=process_id,
                    first_config=first_config,
                    checkpoint_root_path=checkpoint_root_path,
                )
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"Process {process_id} (first training) encountered an error:\n{e}")
                logging.error(tb)
                try:
                    error_queue.put((process_id, tb))
                except Exception:
                    logging.error("Failed to put exception info into error queue (first training).")
                os._exit(1)

    @staticmethod
    def _first_training_impl(process_id: int, first_config: str, checkpoint_root_path: Path):
        gpt2_7_steps_config_file_path = working_dir / first_config
        gpt2_7_steps_config_dict = load_app_config_dict(gpt2_7_steps_config_file_path, experiment_id="0")

        checkpoint_path = str(checkpoint_root_path)
        gpt2_7_steps_config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
            "checkpoint_path"
        ] = checkpoint_path
        gpt2_7_steps_config_dict["settings"]["paths"]["checkpoint_saving_path"] = checkpoint_path
        loss_values_experiment_0_path = checkpoint_root_path / "experiment_0_loss_scores.txt"
        scheduler_info_path = checkpoint_root_path / "experiment_0_scheduler_info.json"

        main_obj_0 = Main(gpt2_7_steps_config_file_path)
        main_obj_0.config_dict = gpt2_7_steps_config_dict
        main_obj_0.add_custom_component(
            component_key="results_subscriber",
            variant_key="save_all",
            custom_component=SaveAllResultSubscriber,
            custom_config=SaveAllResultSubscriberConfig,
        )
        components_0: TrainingComponentsInstantiationModel = main_obj_0.build_components(
            components_model_type=TrainingComponentsInstantiationModel
        )
        main_obj_0.run(components_0)

        # we collect the loss values from rank 0 and store them in the temporary experiment folder
        if dist.get_rank() == 0:
            messages_0: list[Message[EvaluationResultBatch]] = components_0.evaluation_subscriber.message_list
            loss_scores_0 = _get_loss_scores(messages_0, "train loss avg")
            with open(loss_values_experiment_0_path, "w") as f:
                json.dump(loss_scores_0, f)

            # make sure that the checkpoints have been written and checkpoint info file has been updated
            checkpoint_info_file_path = checkpoint_root_path / "0" / "last_checkpoint_info.json"
            assert checkpoint_info_file_path.exists(), "Missing last_checkpoint_info.json after first training."
            with open(checkpoint_info_file_path, "r") as f:
                checkpoint_info = json.load(f)
            expected_cp_suffix = "eid_0-seen_steps_4-seen_tokens_4096-target_steps_7-target_tokens_7168"
            assert checkpoint_info["checkpoint_folder_path"].endswith(
                expected_cp_suffix
            ), "Checkpoint info file does not point to expected step 4 folder."
            assert Path(checkpoint_info["checkpoint_folder_path"]).exists(), "Checkpoint folder path does not exist."

            # enumerate checkpoint paths and ensure max seen matches info
            checkpoint_paths = list(checkpoint_root_path.glob("**/*seen_steps_*-seen_tokens_*"))
            assert checkpoint_paths, "No checkpoint folders found."
            max_seen_steps = -1
            max_seen_tokens = -1
            for cp in checkpoint_paths:
                seen_steps, seen_tokens = _extract_seen_steps_and_tokens(cp.name)
                max_seen_steps = max(max_seen_steps, seen_steps)
                max_seen_tokens = max(max_seen_tokens, seen_tokens)
            cp_info_seen_steps, cp_info_seen_tokens = _extract_seen_steps_and_tokens(
                Path(checkpoint_info["checkpoint_folder_path"]).name
            )
            assert cp_info_seen_steps == max_seen_steps, "Checkpoint info seen_steps not max."
            assert cp_info_seen_tokens == max_seen_tokens, "Checkpoint info seen_tokens not max."

            # store scheduler info needed for second run assertions
            lr_scheduler_0 = components_0.app_state.lr_scheduler
            scheduler_info: dict[str, Any] = {
                "base_lrs": lr_scheduler_0.base_lrs,
                "last_epoch": lr_scheduler_0.last_epoch,
                "last_lr": lr_scheduler_0.get_last_lr(),
            }
            with open(scheduler_info_path, "w") as f:
                json.dump(scheduler_info, f)

    # ---------- Second (warm start) training wrappers ----------

    @staticmethod
    def _second_training_impl_wrapper(
        process_id: int,
        world_size_second: int,
        second_config: str,
        checkpoint_root_path: Path,
        error_queue: Any,
    ):
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size_second,
            rdvz_port=24572,
        ):
            try:
                TestWarmstart._second_training_impl(
                    process_id=process_id,
                    second_config=second_config,
                    checkpoint_root_path=checkpoint_root_path,
                )
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"Process {process_id} (second training) encountered an error:\n{e}")
                logging.error(tb)
                try:
                    error_queue.put((process_id, tb))
                except Exception:
                    logging.error("Failed to put exception info into error queue (second training).")
                os._exit(1)

    @staticmethod
    def _second_training_impl(process_id: int, second_config: str, checkpoint_root_path: Path):
        gpt2_warm_start_config_file_path = working_dir / second_config
        gpt2_warm_start_config_dict = load_app_config_dict(gpt2_warm_start_config_file_path, experiment_id="1")

        checkpoint_path = str(checkpoint_root_path)
        # path to checkpoint from first training (step 4)
        warmstart_checkpoint_dir = (
            checkpoint_root_path / "0" / "eid_0-seen_steps_4-seen_tokens_4096-target_steps_7-target_tokens_7168"
        )
        gpt2_warm_start_config_dict["app_state"]["config"]["checkpoint_dir_path"] = str(warmstart_checkpoint_dir)
        gpt2_warm_start_config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
            "checkpoint_path"
        ] = checkpoint_path
        gpt2_warm_start_config_dict["settings"]["paths"]["checkpoint_saving_path"] = checkpoint_path
        # loss_values_experiment_1_path = checkpoint_root_path / "experiment_1_loss_scores.txt"
        scheduler_info_path = checkpoint_root_path / "experiment_0_scheduler_info.json"

        main_obj_1 = Main(gpt2_warm_start_config_file_path)
        main_obj_1.config_dict = gpt2_warm_start_config_dict
        main_obj_1.add_custom_component(
            component_key="results_subscriber",
            variant_key="save_all",
            custom_component=SaveAllResultSubscriber,
            custom_config=SaveAllResultSubscriberConfig,
        )
        components_1: TrainingComponentsInstantiationModel = main_obj_1.build_components(
            components_model_type=TrainingComponentsInstantiationModel
        )

        # if dist.get_rank() == 0:
        # load scheduler info from first training
        with open(scheduler_info_path, "r") as f:
            scheduler_info = json.load(f)
        lr_scheduler = components_1.app_state.lr_scheduler
        assert (
            scheduler_info["base_lrs"] == lr_scheduler.base_lrs
        ), "Initial base_lrs mismatch between first and warmstart trainings."
        assert lr_scheduler.last_epoch == 4, "Warmstart scheduler must start at epoch 4."

        main_obj_1.run(components_1)

        if dist.get_rank() == 0:
            messages_1: list[Message[EvaluationResultBatch]] = components_1.evaluation_subscriber.message_list
            loss_scores_1 = _get_loss_scores(messages_1, "train loss avg")

            with open(checkpoint_root_path / "experiment_0_loss_scores.txt", "r") as f:
                loaded_loss_values_0 = json.load(f)
            assert loaded_loss_values_0[4:] == pytest.approx(
                loss_scores_1, abs=1e-16
            ), "Warmstart loss trajectory mismatch with from-scratch continuation."

            # Additionally assert checkpoint info integrity from first run still present
            checkpoint_info_file_path = checkpoint_root_path / "0" / "last_checkpoint_info.json"
            assert checkpoint_info_file_path.exists(), "Missing last_checkpoint_info.json from first training."
            with open(checkpoint_info_file_path, "r") as f:
                checkpoint_info = json.load(f)
            assert checkpoint_info["checkpoint_folder_path"].endswith(
                "eid_0-seen_steps_4-seen_tokens_4096-target_steps_7-target_tokens_7168"
            ), "Incorrect checkpoint folder path recorded."

        # Compare final scheduler state
        with open(scheduler_info_path, "r") as f:
            scheduler_info = json.load(f)
        lr_scheduler = components_1.app_state.lr_scheduler
        assert (
            lr_scheduler.last_epoch == scheduler_info["last_epoch"]
        ), "Scheduler last_epoch mismatch after warmstart training."  # both should reach same final epoch
        assert (
            lr_scheduler.get_last_lr() == scheduler_info["last_lr"]
        ), "Scheduler last_lr mismatch after warmstart training."

    # ---------- Dataloader warmstart test (multiprocessing) ----------

    def test_warmstart_dataloader(self):
        world_size = 8
        manager = py_mp.Manager()
        error_queue = manager.Queue()
        proc_ctx = mp.spawn(
            TestWarmstart._dataloader_test_impl_wrapper,
            args=(world_size, error_queue),
            nprocs=world_size,
            join=False,
        )
        TestWarmstart._monitor_child_processes(manager, error_queue, proc_ctx)

    @staticmethod
    def _dataloader_test_impl_wrapper(process_id: int, world_size: int, error_queue: Any):
        non_skipped_cfg = working_dir / "gpt2_train_num_steps_7_pp_tp.yaml"
        skipped_cfg = working_dir / "gpt2_warm_start_from_step_4_pp_tp.yaml"
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=24573,
        ):
            try:
                TestWarmstart._dataloader_test_impl(non_skipped_cfg, skipped_cfg)
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"Process {process_id} (dataloader test) encountered an error:\n{e}")
                logging.error(tb)
                try:
                    error_queue.put((process_id, tb))
                except Exception:
                    logging.error("Failed to put exception info into error queue (dataloader test).")
                os._exit(1)

    @staticmethod
    def _dataloader_test_impl(non_skipped_cfg_path: Path, skipped_cfg_path: Path):
        gpt2_two_steps_config_dict = load_app_config_dict(non_skipped_cfg_path, experiment_id="0")
        gpt2_warm_start_from_step_4_dict = load_app_config_dict(skipped_cfg_path, experiment_id="1")

        main_obj_1 = Main(non_skipped_cfg_path)
        main_obj_1.config_dict = gpt2_two_steps_config_dict

        main_obj_2 = Main(skipped_cfg_path)
        main_obj_2.config_dict = gpt2_warm_start_from_step_4_dict

        main_obj_1.add_custom_component(
            component_key="results_subscriber",
            variant_key="save_all",
            custom_component=SaveAllResultSubscriber,
            custom_config=SaveAllResultSubscriberConfig,
        )
        components_1: TrainDataloaderInstantiationModel = main_obj_1.build_components(
            components_model_type=TrainDataloaderInstantiationModel
        )
        dataloader_1: LLMDataLoader = components_1.train_dataloader
        dl_1_samples = [s for s in dataloader_1]

        main_obj_2.add_custom_component(
            component_key="results_subscriber",
            variant_key="save_all",
            custom_component=SaveAllResultSubscriber,
            custom_config=SaveAllResultSubscriberConfig,
        )
        components_2 = main_obj_2.build_components(components_model_type=TrainDataloaderInstantiationModel)
        dataloader_2: LLMDataLoader = components_2.train_dataloader
        dl_2_samples = [s for s in dataloader_2]

        num_skip_steps: int = components_2.settings.training_progress.num_seen_steps
        assert num_skip_steps == 4, "Warmstart dataloader must skip 4 steps"
        assert len(dl_1_samples) == num_skip_steps + len(dl_2_samples), "Sample length mismatch after skipping"
        assert components_1.settings.training_progress.num_seen_steps == 0, "First dataloader should not skip steps"

        # iterate through both sample lists from the dataloaders
        # and assert the equality of the samples
        for i in range(len(dataloader_2)):
            assert (
                dl_1_samples[i + num_skip_steps].samples["input_ids"].equal(dl_2_samples[i].samples["input_ids"])
            ), "Sample equality check failed after skip adjustment"
            # mutate one tensor to ensure inequality check triggers
            dl_1_samples[i + num_skip_steps].samples["input_ids"][-1] = 0
            assert (
                not dl_1_samples[i + num_skip_steps].samples["input_ids"].equal(dl_2_samples[i].samples["input_ids"])
            ), "Mutation should have broken tensor equality"

    # -------- Multiprocessing helpers ---------

    @staticmethod
    def _monitor_child_processes(manager: Any, error_queue: Any, proc_ctx: Any) -> None:
        """Monitors spawned child processes and terminates remaining workers early if any child reports an exception.

        Copied (with tiny adaptations) from other multiprocessing test utilities in the repository.
        """
        processes = []
        if proc_ctx is None:
            processes = []
        else:
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
                try:
                    processes = list(proc_ctx)
                except Exception:
                    if hasattr(proc_ctx, "terminate") or hasattr(proc_ctx, "is_alive") or hasattr(proc_ctx, "join"):
                        processes = [proc_ctx]
                    else:
                        processes = []

        try:
            while True:
                if not error_queue.empty():
                    proc_id, tb = error_queue.get()
                    for p in processes:
                        try:
                            alive = p.is_alive() if hasattr(p, "is_alive") else True
                            if alive and hasattr(p, "terminate"):
                                p.terminate()
                        except Exception:
                            pass
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

                all_finished = all((not p.is_alive()) for p in processes)
                if all_finished:
                    for p in processes:
                        try:
                            p.join()
                        except Exception:
                            pass
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


def _get_loss_scores(messages: list[Message[EvaluationResultBatch]], loss_key: str) -> list[float]:
    return [message.payload.losses[loss_key].value.item() for message in messages]


def _extract_seen_steps_and_tokens(filename: str) -> tuple[int, int]:
    pattern = r"seen_steps_(\d+)-seen_tokens_(\d+)"
    match = re.search(pattern, filename)
    if match is None:
        raise ValueError(f"Filename '{filename}' does not match expected pattern '{pattern}'.")
    return int(match.group(1)), int(match.group(2))
