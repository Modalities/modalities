import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modalities.__main__ import Main, load_app_config_dict
from modalities.batch import EvaluationResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.logging_broker.messages import Message
from tests.end2end_tests.custom_components import (
    MultiProcessingCudaEnv,
    SaveAllResultSubscriber,
    SaveAllResultSubscriberConfig,
)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs.",
)
class TestContextParallelParity:
    @staticmethod
    def _build_config_dict(base_config_path: Path, experiments_root_path: Path, experiment_id: str, cp_enabled: bool):
        config_dict = load_app_config_dict(
            config_file_path=base_config_path,
            experiments_root_path=experiments_root_path,
            experiment_id=experiment_id,
        )

        # Keep runs tiny and deterministic for parity checks.
        config_dict["settings"]["intervals"]["training_log_interval_in_steps"] = 1
        config_dict["settings"]["intervals"]["checkpointing_interval_in_steps"] = 4
        config_dict["settings"]["intervals"]["evaluation_interval_in_steps"] = 4
        config_dict["settings"]["step_profile"]["gradient_accumulation_steps"] = 1
        config_dict["settings"]["step_profile"]["local_train_micro_batch_size"] = 1
        config_dict["settings"]["step_profile"]["sequence_length"] = 256

        # Use SDPA backend in both runs to isolate CP impact.
        config_dict["model_raw"]["config"]["attention_implementation"] = "pytorch_flash"

        # Remove conversion components and use explicit target settings for stable tiny runs.
        config_dict["settings"]["training_target"]["num_target_steps"] = 4
        if cp_enabled:
            # dp_degree becomes 1 when cp=2 and world_size=2.
            config_dict["settings"]["training_target"]["num_target_tokens"] = 1024
            config_dict["device_mesh"]["config"]["context_parallel_degree"] = 2
            config_dict["device_mesh"]["config"]["data_parallel_shard_degree"] = -1

            config_dict["gpt2_cp_model"] = {
                "component_key": "model",
                "variant_key": "gpt2_cp",
                "config": {
                    "model": {"instance_key": "model_raw", "pass_type": "BY_REFERENCE"},
                    "device_mesh": {"instance_key": "device_mesh", "pass_type": "BY_REFERENCE"},
                    "context_parallel_load_balancer": "headtail",
                },
            }
            config_dict["fsdp_model"]["config"]["model"] = {
                "instance_key": "gpt2_cp_model",
                "pass_type": "BY_REFERENCE",
            }

            sampler_cfg = config_dict["train_dataloader"]["config"]["batch_sampler"]["config"]["sampler"]
            sampler_cfg["variant_key"] = "resumable_distributed_multi_dim_sampler"
            sampler_cfg["config"] = {
                "dataset": {"instance_key": "train_dataset", "pass_type": "BY_REFERENCE"},
                "device_mesh": {"instance_key": "device_mesh", "pass_type": "BY_REFERENCE"},
                "data_parallel_key": "dp_shard",
                "shuffle": True,
                "seed": 42,
                "drop_last": True,
                "skip_num_global_samples": 0,
            }
        else:
            # dp_degree is world_size=2 in non-CP run.
            config_dict["settings"]["training_target"]["num_target_tokens"] = 2048
            config_dict["device_mesh"]["config"]["context_parallel_degree"] = 1
            config_dict["fsdp_model"]["config"]["model"] = {
                "instance_key": "model_raw",
                "pass_type": "BY_REFERENCE",
            }
            config_dict.pop("gpt2_cp_model", None)

        return config_dict

    @staticmethod
    def _run_training_and_write_losses(
        process_id: int,
        world_size: int,
        rdvz_port: int,
        base_config_path: Path,
        experiments_root_path: Path,
        experiment_id: str,
        cp_enabled: bool,
        output_path: Path,
    ):
        torch.manual_seed(20)
        torch.cuda.manual_seed(20)

        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
        ):
            config_dict = TestContextParallelParity._build_config_dict(
                base_config_path=base_config_path,
                experiments_root_path=experiments_root_path,
                experiment_id=experiment_id,
                cp_enabled=cp_enabled,
            )

            main_obj = Main(
                base_config_path,
                experiments_root_path=experiments_root_path,
                experiment_id=experiment_id,
            )
            main_obj.config_dict = config_dict
            main_obj.add_custom_component(
                component_key="results_subscriber",
                variant_key="save_all",
                custom_component=SaveAllResultSubscriber,
                custom_config=SaveAllResultSubscriberConfig,
            )

            components: TrainingComponentsInstantiationModel = main_obj.build_components(
                components_model_type=TrainingComponentsInstantiationModel
            )
            main_obj.run(components)

            if dist.get_rank() == 0:
                messages: list[Message[EvaluationResultBatch]] = components.evaluation_subscriber.message_list
                losses = [float(m.payload.losses["train loss avg"].value) for m in messages]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(losses, f)

    @staticmethod
    def test_cp_vs_non_cp_loss_parity(tmp_path: Path):
        working_dir = Path(os.path.dirname(__file__))
        base_config_path = working_dir / "configs" / "fsdp2_gpt2_train_num_steps_8.yaml"

        non_cp_losses_path = tmp_path / "non_cp_losses.json"
        cp_losses_path = tmp_path / "cp_losses.json"

        world_size = 2

        mp.spawn(
            TestContextParallelParity._run_training_and_write_losses,
        from tests.utility import find_free_port
        non_cp_port = find_free_port()
        mp.spawn(
            TestContextParallelParity._run_training_and_write_losses,
            args=(
                world_size,
                non_cp_port,
                base_config_path,
                tmp_path,
                "parity_non_cp",
                False,
                non_cp_losses_path,
            ),
            nprocs=world_size,
            join=True,
        )

        mp.spawn(
            TestContextParallelParity._run_training_and_write_losses,
            args=(
                world_size,
                24832,
                base_config_path,
                tmp_path,
                "parity_cp",
                True,
                cp_losses_path,
            ),
            nprocs=world_size,
            join=True,
        )

        with open(non_cp_losses_path, "r", encoding="utf-8") as f:
            non_cp_losses = json.load(f)
        with open(cp_losses_path, "r", encoding="utf-8") as f:
            cp_losses = json.load(f)

        assert len(non_cp_losses) >= 4
        assert len(cp_losses) >= 4

        # Compare aligned prefix for parity signal.
        n = min(len(non_cp_losses), len(cp_losses))
        diffs = [abs(non_cp_losses[i] - cp_losses[i]) for i in range(n)]

        # We allow moderate drift; this is a regression guard against severe mismatch.
        assert diffs[-1] < 0.5, f"Final loss diverged too much: {diffs[-1]}"
        assert (sum(diffs) / len(diffs)) < 0.35, f"Average loss delta too high: {sum(diffs) / len(diffs)}"
