import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
import torch
import torch.distributed as dist
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer

from modalities.__main__ import load_app_config_dict
from modalities.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2LLMConfig
from modalities.running_env.cuda_env import FSDPRunningEnv, FSDPRunningEnvConfig, RunningEnv

# NOTE: We need to run the tests in a torch distributed environment with at least two GPUs.
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 \
#   /path/to/pytest path/to/test_fsdp_to_disc_checkpointing.py

_ROOT_DIR = Path(__file__).parents[1]


class ExperimentConfig(BaseModel):
    llm_model_conf: GPT2LLMConfig  # Named it llm_model_conf as model_ is a protected namespace in pydantic
    running_env_conf: FSDPRunningEnvConfig


@pytest.mark.skip(
    reason="Need to fix absolute path for config_file_path and needs to be run via "
    "torchrun in a torch distributed environment (torchrun)"
)
class TestFSDPToDiscCheckpointing:
    @pytest.fixture
    def experiment_config(self) -> ExperimentConfig:
        config_file_path = _ROOT_DIR / Path("tests/checkpointing/gpt2_config.yaml")
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
            yield Path(tmp_dir_path)

    @staticmethod
    def _generate_batch(experiment_config: ExperimentConfig):
        # prepare input and targets
        data = torch.randint(
            0, experiment_config.llm_model_conf.vocab_size, (8, experiment_config.llm_model_conf.block_size + 1)
        ).cuda()
        batch_input_ids_dict = {experiment_config.llm_model_conf.sample_key: data[:, :-1]}
        batch_target_ids = data[:, 1:]
        batch_target_ids = batch_target_ids.contiguous()
        return batch_input_ids_dict, batch_target_ids

    @staticmethod
    def _forward_backward_pass(
        experiment_config: ExperimentConfig,
        model: FSDP,
        optimizer: Optimizer,
        batch_input_ids_dict: Dict,
        batch_target_ids: torch.Tensor,
    ):
        ce_loss = CrossEntropyLoss()

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
        return loss

    @staticmethod
    def _assert_equality_optimizer_param_group(
        optimizer_1_state_dict: Dict, optimizer_2_state_dict: Dict, must_be_equal: bool
    ):
        if must_be_equal:
            assert optimizer_1_state_dict["param_groups"] == optimizer_2_state_dict["param_groups"]
        else:
            assert not (optimizer_1_state_dict["param_groups"] == optimizer_2_state_dict["param_groups"])

    @staticmethod
    def _assert_equality_optimizer_state(
        optimizer_1_state_dict: Dict, optimizer_2_state_dict: Dict, must_be_equal: bool
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
                    assert torch.equal(state_1[state_key], state_2[state_key])
                else:
                    assert not torch.equal(state_1[state_key], state_2[state_key])

    @staticmethod
    def _assert_equality_two_models(params_1, params_2, must_be_equal: bool):
        for p1, p2 in zip(params_1, params_2):
            if must_be_equal:
                assert torch.equal(p1, p2)
            else:
                assert not torch.equal(p1, p2)

    def test_save_checkpoint_after_backward_pass(
        self,
        fsdp_wrapped_model: FSDP,
        optimizer: Optimizer,
        temporary_checkpoint_folder_path: Path,
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
        )

        untrained_model_parameters = [p.clone() for p in fsdp_wrapped_model.parameters()]
        untrained_optimizer_state_dict = deepcopy(optimizer.state_dict())

        # run backward pass
        batch_input_ids_dict, batch_target_ids = self._generate_batch(experiment_config)
        self._forward_backward_pass(
            experiment_config=experiment_config,
            model=fsdp_wrapped_model,
            optimizer=optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )
        updated_model_parameters = [p.clone() for p in fsdp_wrapped_model.parameters()]
        updated_optimizer_state_dict = deepcopy(optimizer.state_dict())

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

        optimizer_2 = AdamW(fsdp_wrapped_model_2.parameters(), lr=0.001)

        checkpointing.load_optimizer_checkpoint(
            optimizer=optimizer_2,
            wrapped_model=fsdp_wrapped_model_2,
            experiment_id=experiment_id,
            global_train_batch_id=global_train_batch_id,
        )

        loaded_and_updated_model_parameters = [p.clone() for p in fsdp_wrapped_model_2.parameters()]
        loaded_and_updated_optimizer_state_dict = deepcopy(optimizer_2.state_dict())

        # make sure that after the update all weights are DIFFERENT from the original ones
        self._assert_equality_two_models(updated_model_parameters, untrained_model_parameters, must_be_equal=False)
        self._assert_equality_optimizer_param_group(
            updated_optimizer_state_dict, untrained_optimizer_state_dict, must_be_equal=True
        )

        # make sure that the updated parameters are EQUAL to the ones that we saved subsequently
        self._assert_equality_two_models(
            updated_model_parameters, loaded_and_updated_model_parameters, must_be_equal=True
        )
        self._assert_equality_optimizer_param_group(
            updated_optimizer_state_dict, loaded_and_updated_optimizer_state_dict, must_be_equal=True
        )
        self._assert_equality_optimizer_state(
            updated_optimizer_state_dict, loaded_and_updated_optimizer_state_dict, must_be_equal=True
        )

        # we do another forward/backward pass and check
        #  if the weights are equally updated for the loaded model as for the not-loaded model
        # run backward pass
        batch_input_ids_dict, batch_target_ids = self._generate_batch(experiment_config)

        loss_1 = self._forward_backward_pass(
            experiment_config=experiment_config,
            model=fsdp_wrapped_model,
            optimizer=optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )
        loss_2 = self._forward_backward_pass(
            experiment_config=experiment_config,
            model=fsdp_wrapped_model_2,
            optimizer=optimizer_2,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )

        assert loss_1 == loss_2

        # make sure that after another update the two models and optimizers are the same
        self._assert_equality_two_models(
            fsdp_wrapped_model.parameters(), fsdp_wrapped_model_2.parameters(), must_be_equal=True
        )
        self._assert_equality_optimizer_param_group(
            optimizer.state_dict(), optimizer_2.state_dict(), must_be_equal=True
        )
        self._assert_equality_optimizer_state(optimizer.state_dict(), optimizer_2.state_dict(), must_be_equal=True)

        # make sure that the weights and state has changed to the previous forward backward pass
        self._assert_equality_two_models(fsdp_wrapped_model.parameters(), updated_model_parameters, must_be_equal=False)
        self._assert_equality_optimizer_param_group(
            optimizer.state_dict(), updated_optimizer_state_dict, must_be_equal=True
        )
        self._assert_equality_optimizer_state(optimizer.state_dict(), updated_optimizer_state_dict, must_be_equal=False)
