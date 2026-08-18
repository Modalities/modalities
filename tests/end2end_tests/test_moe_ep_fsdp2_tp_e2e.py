import logging
import multiprocessing as py_mp
import os
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from modalities.__main__ import Main, load_app_config_dict
from modalities.batch import EvaluationResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.logging_broker.messages import Message
from tests.end2end_tests.checkpoint_resume_utils import (
    assert_checkpoint_resumes_optimizer_and_scheduler_state,
    get_last_checkpoint_dir_path,
)
from tests.end2end_tests.custom_components import (
    MultiProcessingCudaEnv,
    SaveAllResultSubscriber,
    SaveAllResultSubscriberConfig,
)
from tests.utility import find_free_port, monitor_child_processes


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This E2E test requires 4 CUDA devices.")
class TestMoEEPFSDP2TPE2E:
    """Verifies that expert parallelism (EP) composes correctly with FSDP2 and tensor parallelism (TP)
    at the same time (world_size=4, tp=2 x ep=2 x dp_shard=1)."""

    @staticmethod
    def _patch_for_short_test_run(config_dict: dict[str, Any], checkpoint_root_path: Path) -> None:
        # Keep runtime short while preserving the full EP + FSDP2 + TP wiring.
        config_dict["settings"]["intervals"]["training_log_interval_in_steps"] = 1
        config_dict["settings"]["intervals"]["checkpointing_interval_in_steps"] = 1
        config_dict["settings"]["intervals"]["evaluation_interval_in_steps"] = 1000

        config_dict["settings"]["step_profile"]["sequence_length"] = 64
        config_dict["settings"]["step_profile"]["local_train_micro_batch_size"] = 1
        config_dict["settings"]["step_profile"]["gradient_accumulation_steps"] = 1

        config_dict["settings"]["training_target"]["num_target_tokens"] = 512
        config_dict["settings"]["training_target"]["num_target_steps"] = 2
        config_dict["lr_scheduler"]["config"]["total_steps"] = 2

        config_dict["train_dataset"]["config"]["sequence_length"] = 64
        config_dict["test_dataset"]["config"]["sequence_length"] = 64
        config_dict["train_dataloader"]["config"]["num_workers"] = 0
        config_dict["test_dataloader"]["config"]["num_workers"] = 0
        config_dict["train_dataloader"]["config"]["pin_memory"] = False
        config_dict["test_dataloader"]["config"]["pin_memory"] = False

        config_dict["settings"]["paths"]["checkpoint_saving_path"] = checkpoint_root_path
        config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
            "checkpoint_path"
        ] = checkpoint_root_path

    @staticmethod
    def _worker_wrapper(
        process_id: int,
        world_size: int,
        rdvz_port: int,
        config_file_path: Path,
        tmp_path: Path,
        error_queue: Any,
    ) -> None:
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
        ):
            try:
                TestMoEEPFSDP2TPE2E._worker_impl(
                    process_id=process_id,
                    config_file_path=config_file_path,
                    tmp_path=tmp_path,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                logging.error(f"Process {process_id} failed: {exc}\n{tb}")
                try:
                    error_queue.put((process_id, tb))
                except Exception:
                    logging.error("Failed to write child exception to queue.")
                os._exit(1)

    @staticmethod
    def _build_main_and_components(
        config_file_path: Path, tmp_path: Path, experiment_id: str, checkpoint_root_path: Path
    ) -> tuple[Main, TrainingComponentsInstantiationModel]:
        """Builds a full, independent set of components (model, optimizer, lr scheduler, ...) from the
        test config. Called a second time to resume the checkpoint into fresh instances."""
        cfg = load_app_config_dict(
            config_file_path=config_file_path, experiments_root_path=tmp_path, experiment_id=experiment_id
        )
        TestMoEEPFSDP2TPE2E._patch_for_short_test_run(cfg, checkpoint_root_path)

        main_obj = Main(config_file_path, experiments_root_path=tmp_path, experiment_id=experiment_id)
        main_obj.config_dict = cfg
        main_obj.add_custom_component(
            component_key="results_subscriber",
            variant_key="save_all",
            custom_component=SaveAllResultSubscriber,
            custom_config=SaveAllResultSubscriberConfig,
        )
        main_obj.config_dict["evaluation_subscriber"]["variant_key"] = "save_all"
        main_obj.config_dict["evaluation_subscriber"]["config"] = {}

        components: TrainingComponentsInstantiationModel = main_obj.build_components(
            components_model_type=TrainingComponentsInstantiationModel
        )
        return main_obj, components

    @staticmethod
    def _worker_impl(process_id: int, config_file_path: Path, tmp_path: Path) -> None:
        experiment_id = "moe-ep-fsdp2-tp-e2e"
        checkpoint_root_path = tmp_path / experiment_id / "checkpoints"
        main_obj, components = TestMoEEPFSDP2TPE2E._build_main_and_components(
            config_file_path=config_file_path,
            tmp_path=tmp_path,
            experiment_id=experiment_id,
            checkpoint_root_path=checkpoint_root_path,
        )

        assert getattr(components.model_raw, "_ep_wrapped", False), "Expected EP wrapping marker on raw model."
        first_layer = next(iter(components.model_raw.layers.values()))
        assert getattr(first_layer.ffn.experts, "_ep_enabled", False), "Expected experts to be EP-enabled."

        main_obj.run(components)

        result_messages: list[Message[EvaluationResultBatch]] = components.evaluation_subscriber.message_list
        assert len(result_messages) > 0, "Expected training messages in evaluation subscriber."
        for message in result_messages:
            loss_value = message.payload.losses["train loss avg"].value
            assert torch.isfinite(loss_value), f"Found non-finite train loss: {loss_value}"

        # Resume the checkpoint into a fresh set of components: the EP optimizer(s) use a custom
        # state-dict layout, so only an actual load proves that the checkpointed optimizer state is
        # readable back (a checkpoint marker file on disk does not).
        checkpoint_dir_path = get_last_checkpoint_dir_path(checkpoint_root_path)
        _, resumed_components = TestMoEEPFSDP2TPE2E._build_main_and_components(
            config_file_path=config_file_path,
            tmp_path=tmp_path,
            experiment_id=experiment_id,
            checkpoint_root_path=checkpoint_root_path,
        )
        assert_checkpoint_resumes_optimizer_and_scheduler_state(
            trained_app_state=components.app_state,
            resumed_app_state=resumed_components.app_state,
            checkpoint_dir_path=checkpoint_dir_path,
            global_rank=process_id,
        )

    @staticmethod
    def test_moe_ep_fsdp2_tp_training_and_checkpointing(tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_file_path = repo_root / "config_files/training/config_lorem_ipsum_long_moe_ep_fsdp2_tp.yaml"

        world_size = 4
        rdvz_port = find_free_port()

        manager = py_mp.Manager()
        error_queue = manager.Queue()
        proc_ctx = mp.spawn(
            TestMoEEPFSDP2TPE2E._worker_wrapper,
            args=(world_size, rdvz_port, config_file_path, tmp_path, error_queue),
            nprocs=world_size,
            join=False,
        )

        monitor_child_processes(manager, error_queue, proc_ctx)
