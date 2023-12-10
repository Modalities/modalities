from pathlib import Path
import tempfile
from typing import Any, Generator
from llm_gym.__main__ import load_app_config_dict
from llm_gym.fsdp.fsdp_running_env import FSDPRunningEnv, FSDPRunningEnvConfig, RunningEnv
from llm_gym.models.gpt2.gpt2_model import GPT2LLM, GPTConfig
from pydantic import BaseModel
import pytest
from torch.optim import Optimizer, AdamW
from llm_gym.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
import torch
from torch.nn import CrossEntropyLoss


class ExperimentConfig(BaseModel):
    llm_model_conf: GPTConfig  # Named it llm_model_conf as model_ is a protected namespace in pydantic
    running_env_conf: FSDPRunningEnvConfig


class TestFSDPToDiscCheckpointing:
    @pytest.fixture
    def experiment_config(self) -> ExperimentConfig:
        config_file_path = Path("/raid/s3/opengptx/max_lue/LLMgym/tests/fixtures/gpt2_config.yaml")
        config_dict = load_app_config_dict(config_file_path=config_file_path)
        experiment_config = ExperimentConfig.model_validate(config_dict)
        return experiment_config

    @pytest.fixture(scope="function")
    def gpt2_model(self, experiment_config: ExperimentConfig) -> GPT2LLM:
        model = GPT2LLM(config=experiment_config.llm_model_conf)
        return model

    @pytest.fixture(scope="function")
    def gpt2_model_2(self, experiment_config: ExperimentConfig) -> GPT2LLM:
        model = GPT2LLM(config=experiment_config.llm_model_conf)
        return model

    @pytest.fixture
    def fsdp_running_env(self, experiment_config: ExperimentConfig) -> Generator[RunningEnv, Any, Any]:
        running_env = FSDPRunningEnv(**dict(experiment_config.running_env_conf))
        with running_env as running_env:
            yield running_env

    @pytest.fixture
    def fsdp_wrapped_model(self, gpt2_model: GPT2LLM, fsdp_running_env) -> FSDP:
        wrapped_model: FSDP = FSDPRunningEnv.wrap_model(gpt2_model, sync_module_states=True)
        return wrapped_model

    @pytest.fixture
    def optimizer(self, fsdp_wrapped_model: GPT2LLM) -> Optimizer:
        optimizer = AdamW(fsdp_wrapped_model.parameters(), lr=0.001)
        return optimizer

    @pytest.fixture
    def temporary_checkpoint_folder_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            yield tmp_dir_path

    @staticmethod
    def _forward_backward_pass(experiment_config: ExperimentConfig, model: FSDP, optimizer: Optimizer):
        ce_loss = CrossEntropyLoss()

        # prepare input and targets
        data = torch.randint(
            0, experiment_config.llm_model_conf.vocab_size, (8, experiment_config.llm_model_conf.block_size + 1)
        ).cuda()
        batch_input_ids_dict = {experiment_config.llm_model_conf.sample_key: data[:, :-1]}
        batch_target_ids = data[:, 1:]
        batch_target_ids = batch_target_ids.contiguous()

        # clear the gradients
        optimizer.zero_grad()

        # forward pass
        predictions = model.forward(inputs=batch_input_ids_dict)[experiment_config.llm_model_conf.prediction_key]
        predictions = predictions.contiguous()
        # backward pass
        loss = ce_loss(predictions.view(-1, predictions.size(-1)), batch_target_ids.view(-1))
        loss.backward()

        # update the weights based on the gradients
        optimizer.step()

    def test_save_checkpoint_after_backward_pass(
        self,
        fsdp_wrapped_model: FSDP,
        optimizer: Optimizer,
        temporary_checkpoint_folder_path: str,
        gpt2_model_2: GPT2LLM,
        experiment_config: ExperimentConfig,
    ):
        experiment_id = "0"
        global_train_batch_id = 1

        checkpointing = FSDPToDiscCheckpointing(
            checkpoint_path=temporary_checkpoint_folder_path,
            experiment_id=experiment_id,
            global_rank=dist.get_rank(),
            model_wrapping_fn=FSDPRunningEnv.wrap_model,
            checkpointing_rank=0,
        )

        untrained_model_parameters = fsdp_wrapped_model.parameters()

        # run backward pass
        self._forward_backward_pass(experiment_config=experiment_config, model=fsdp_wrapped_model, optimizer=optimizer)
        updated_model_parameters = fsdp_wrapped_model.parameters()

        # save model and optimizer before backward pass
        checkpointing._save_checkpoint(
            model=fsdp_wrapped_model, optimizer=optimizer, global_train_batch_id=global_train_batch_id
        )

        # load the model checkpoint
        fsdp_wrapped_model_2 = checkpointing.load_model_checkpoint(
            model=gpt2_model_2,
            experiment_id=experiment_id,
            global_train_batch_id=global_train_batch_id,
        )
        # checkpointing.load_optimizer_checkpoint(
        #     optimizer=optimizer,
        #     model=fsdp_wrapped_model_2,
        #     experiment_id=experiment_id,
        #     global_train_batch_id=global_train_batch_id,
        # )
        loaded_and_updated_model_parameters = fsdp_wrapped_model_2.parameters()

        # make sure that after the update all weights are different from the original ones
        for p1, p2 in zip(updated_model_parameters, untrained_model_parameters):
            assert torch.all(p1 != p2)
        # make sure that the updated parameters are equal to the ones that we saved subsequently
        for p1, p2 in zip(updated_model_parameters, loaded_and_updated_model_parameters):
            assert torch.all(p1 == p2)

        # we should do another forward/backward pass and check if the weights are equally updated for the loaded model as for the not-loded model
