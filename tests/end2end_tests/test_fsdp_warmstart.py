import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from pydantic import BaseModel

from modalities.__main__ import Main, load_app_config_dict
from modalities.batch import EvaluationResultBatch
from modalities.config.config import ProcessGroupBackendType, PydanticLLMDataLoaderIFType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import Message
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.running_env.cuda_env import CudaEnv


def extract_seen_steps_and_tokens(filename: str) -> tuple[int, int]:
    pattern = r"seen_steps_(\d+)-seen_tokens_(\d+)"
    match = re.search(pattern, filename)
    return int(match.group(1)), int(match.group(2))


# NOTE: We need to run the tests in a torch distributed environment with at least two GPUs.
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 \
#   $(which pytest) path/to/test_fsdp_to_disc_checkpointing.py

# NOTE that we can only run one test at time due to NCCL issues with multiple tests in parallel.
# You can specify the test to run with the -k flag, e.g.: -k test_warm_start


working_dir = Path(os.path.dirname(__file__))


class SaveAllResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    def __init__(self):
        self.message_list: list[Message[EvaluationResultBatch]] = []

    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        self.message_list.append(message)

    def consume_dict(self, mesasge_dict: dict[str, Any]):
        pass


class SaveAllResultSubscriberConfig(BaseModel):
    pass


class TrainDataloaderInstantiationModel(BaseModel):
    settings: TrainingComponentsInstantiationModel.Settings
    train_dataloader: PydanticLLMDataLoaderIFType


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
class TestWarmstart:
    @staticmethod
    def get_loss_scores(messages: list[Message[EvaluationResultBatch]], loss_key: str) -> list[float]:
        return [message.payload.losses[loss_key].value.item() for message in messages]

    def test_warm_start(self):
        # We want to verify that the training continues after starting from checkpoint (i.e, warm start)
        # exactly the same way, as if we trained it from scratch.
        # To do so, we have two confings. The first config trains a model for 8 steps and
        # saves multiple intermediary checkpoints.
        # The second config starts from the 4th step and trains the model for 4 more steps.
        # We compare the loss values of the two models after 4 steps and expect them to be the same.

        with tempfile.TemporaryDirectory() as temp_dir:
            # config for two steps model
            gpt2_8_steps_config_file_path = working_dir / "gpt2_train_num_steps_8.yaml"
            gpt2_8_steps_config_dict = load_app_config_dict(gpt2_8_steps_config_file_path, experiment_id="0")

            # adopt the checkpoint path
            checkpoint_path = temp_dir
            gpt2_8_steps_config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
                "checkpoint_path"
            ] = checkpoint_path
            gpt2_8_steps_config_dict["settings"]["paths"]["checkpointing_path"] = checkpoint_path
            loss_values_experiment_0_path = checkpoint_path + "/experiment_0_loss_scores.txt"

            # config for one step model
            gpt2_warm_start_after_4_steps_config_file_path = working_dir / "gpt2_warm_start_from_step_4.yaml"
            gpt2_warm_start_after_4_steps_dict = load_app_config_dict(
                gpt2_warm_start_after_4_steps_config_file_path, experiment_id="1"
            )

            # adopt the checkpoint path
            gpt2_warm_start_after_4_steps_dict["wrapped_model"]["config"]["checkpoint_path"] = (
                checkpoint_path + "/0/eid_0-model-seen_steps_4-seen_tokens_2048-target_steps_15-target_tokens_7680.bin"
            )
            gpt2_warm_start_after_4_steps_dict["optimizer"]["config"]["checkpoint_path"] = (
                checkpoint_path
                + "/0/eid_0-optimizer-seen_steps_4-seen_tokens_2048-target_steps_15-target_tokens_7680.bin"
            )
            gpt2_warm_start_after_4_steps_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
                "checkpoint_path"
            ] = checkpoint_path
            gpt2_warm_start_after_4_steps_dict["settings"]["paths"]["checkpointing_path"] = checkpoint_path
            loss_values_experiment_1_path = checkpoint_path + "/experiment_1_loss_scores.txt"

            # # adopt dataset path
            # gpt2_warm_start_after_4_steps_dict["train_dataset"]["config"]["raw_data_path"] = (
            #     working_dir / "lorem_ipsum.pbin"
            # )

            with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
                main_obj_0 = Main(gpt2_8_steps_config_file_path)
                main_obj_0.config_dict = gpt2_8_steps_config_dict
                main_obj_0.add_custom_component(
                    component_key="results_subscriber",
                    variant_key="save_all",
                    custom_component=SaveAllResultSubscriber,
                    custom_config=SaveAllResultSubscriberConfig,
                )
                print(
                    main_obj_0.config_dict["settings"]["training_target"]["num_target_tokens"]["config"]["dataset_path"]
                )
                components_0 = main_obj_0.build_components(components_model_type=TrainingComponentsInstantiationModel)
                main_obj_0.run(components_0)

                # we collect the loss values from rank 0 and store them in the temporary experiment folder
                if dist.get_rank() == 0:
                    messages_0: list[Message[EvaluationResultBatch]] = components_0.evaluation_subscriber.message_list
                    loss_scores_0 = TestWarmstart.get_loss_scores(messages_0, "train loss avg")
                    with open(loss_values_experiment_0_path, "w") as f:
                        json.dump(loss_scores_0, f)

                    # make sure that the checkpoints have been written and checkpoint info file has been updated
                    checkpoint_info_file_path = Path(checkpoint_path) / "0/last_checkpoint_info.json"
                    assert checkpoint_info_file_path.exists()
                    with open(checkpoint_info_file_path, "r") as f:
                        checkpoint_info = json.load(f)
                    assert checkpoint_info["model_checkpoint_path"] == (
                        checkpoint_path
                        + "/0/eid_0-model-seen_steps_12-seen_tokens_6144-target_steps_15-target_tokens_7680.bin"
                    )
                    assert checkpoint_info["optimizer_checkpoint_path"] == (
                        checkpoint_path
                        + "/0/eid_0-optimizer-seen_steps_12-seen_tokens_6144-target_steps_15-target_tokens_7680.bin"
                    )
                    assert Path(checkpoint_info["model_checkpoint_path"]).exists()
                    assert Path(checkpoint_info["optimizer_checkpoint_path"]).exists()

                    checkpoint_paths = list(Path(checkpoint_path).glob("**/*.bin"))
                    model_max_seen_steps = -1
                    model_max_seen_tokens = -1
                    optimizer_max_seen_steps = -1
                    optimizer_max_seen_tokens = -1
                    for checkpoint_path in checkpoint_paths:
                        seen_steps, seen_tokens = extract_seen_steps_and_tokens(checkpoint_path.name)
                        if "model" in checkpoint_path.name:
                            model_max_seen_steps = max(model_max_seen_steps, seen_steps)
                            model_max_seen_tokens = max(model_max_seen_tokens, seen_tokens)
                        elif "optimizer" in checkpoint_path.name:
                            optimizer_max_seen_steps = max(optimizer_max_seen_steps, seen_steps)
                            optimizer_max_seen_tokens = max(optimizer_max_seen_tokens, seen_tokens)
                        else:
                            raise ValueError("Invalid checkpoint path")

                    assert model_max_seen_steps == optimizer_max_seen_steps
                    assert model_max_seen_tokens == optimizer_max_seen_tokens

                    cp_info_model_seen_steps, cp_info_model_seen_tokens = extract_seen_steps_and_tokens(
                        checkpoint_info["model_checkpoint_path"]
                    )
                    cp_info_optimizer_seen_steps, cp_info_optimizer_seen_tokens = extract_seen_steps_and_tokens(
                        checkpoint_info["optimizer_checkpoint_path"]
                    )
                    assert cp_info_model_seen_steps == cp_info_optimizer_seen_steps
                    assert cp_info_model_seen_tokens == cp_info_optimizer_seen_tokens

                    assert cp_info_model_seen_steps == model_max_seen_steps
                    assert cp_info_model_seen_tokens == model_max_seen_tokens

                main_obj_1 = Main(gpt2_warm_start_after_4_steps_config_file_path)
                main_obj_1.config_dict = gpt2_warm_start_after_4_steps_dict

                main_obj_1.add_custom_component(
                    component_key="results_subscriber",
                    variant_key="save_all",
                    custom_component=SaveAllResultSubscriber,
                    custom_config=SaveAllResultSubscriberConfig,
                )
                components_1 = main_obj_1.build_components(components_model_type=TrainingComponentsInstantiationModel)

                assert (
                    components_0.app_state.lr_scheduler.base_lrs == components_1.app_state.lr_scheduler.base_lrs
                )  # make sure that the initial learning rates are the same
                assert components_1.app_state.lr_scheduler.last_epoch == 4  # we start from step 4

                main_obj_1.run(components_1)

                # we collect the loss values from rank 0 for the warmstart model
                # and store them in the temporary experiment folder
                if dist.get_rank() == 0:
                    messages_1: list[Message[EvaluationResultBatch]] = components_1.evaluation_subscriber.message_list
                    loss_scores_1 = TestWarmstart.get_loss_scores(messages_1, "train loss avg")
                    with open(loss_values_experiment_1_path, "w") as f:
                        json.dump(loss_scores_1, f)

                    # read the losses from disc
                    # note that the temporary directory is only correct for the rank 0.
                    # rank 1 has a different one and we don't store anything in there
                    with open(loss_values_experiment_0_path, "r") as f:
                        loaded_loss_values_0 = json.load(f)

                    with open(loss_values_experiment_1_path, "r") as f:
                        loaded_loss_values_1 = json.load(f)

                    # we check if the losses for the model from scratch
                    # and the warm start model have the same loss values
                    assert loaded_loss_values_0[4:] == pytest.approx(loaded_loss_values_1, abs=1e-16)

                # assert that the scheduler state is the same for both models
                assert components_1.app_state.lr_scheduler.last_epoch == components_0.app_state.lr_scheduler.last_epoch
                assert (
                    components_0.app_state.lr_scheduler.get_last_lr()
                    == components_1.app_state.lr_scheduler.get_last_lr()
                )

    def test_warmstart_dataloader(self):
        # non-skipped config
        gpt2_two_steps_config_file_path = working_dir / "gpt2_train_num_steps_8.yaml"
        gpt2_two_steps_config_dict = load_app_config_dict(gpt2_two_steps_config_file_path, experiment_id="0")

        # skipped config
        gpt2_warm_start_from_step_1_config_file_path = working_dir / "gpt2_warm_start_from_step_4.yaml"
        gpt2_warm_start_from_step_1_dict = load_app_config_dict(
            gpt2_warm_start_from_step_1_config_file_path, experiment_id="1"
        )

        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            main_obj_1 = Main(gpt2_two_steps_config_file_path)
            main_obj_1.config_dict = gpt2_two_steps_config_dict

            main_obj_2 = Main(gpt2_warm_start_from_step_1_config_file_path)
            main_obj_2.config_dict = gpt2_warm_start_from_step_1_dict

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

            # fast forward the first dataloader

            num_skip_steps = components_2.settings.training_progress.num_seen_steps

            # make sure that we actually skip as defined in the config
            assert num_skip_steps == 4
            assert len(dl_1_samples) == num_skip_steps + len(dl_2_samples)

            # make sure that the first dataloader is not skipped
            assert components_1.settings.training_progress.num_seen_steps == 0

            # iterate through both sample lists from the dataloaders
            # and assert the equality of the samples

            for i in range(len(dataloader_2)):
                assert dl_1_samples[i + num_skip_steps].samples["input_ids"].equal(dl_2_samples[i].samples["input_ids"])

                dl_1_samples[i + num_skip_steps].samples["input_ids"][-1] = 0
                assert not (
                    dl_1_samples[i + num_skip_steps].samples["input_ids"].equal(dl_2_samples[i].samples["input_ids"])
                )
