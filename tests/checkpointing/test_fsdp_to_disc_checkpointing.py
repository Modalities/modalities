import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Optimizer

from modalities.__main__ import load_app_config_dict
from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import FSDPCheckpointLoading
from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import CheckpointingEntityType, FSDPCheckpointSaving
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2LLMConfig
from modalities.models.model_factory import ModelFactory
from modalities.optimizers.optimizer_factory import OptimizerFactory
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.utils.number_conversion import NumberConversion

# NOTE: We need to run the tests in a torch distributed environment with at least two GPUs.
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 \
#   $(which pytest) path/to/test_fsdp_to_disc_checkpointing.py


_ROOT_DIR = Path(__file__).parents[1]
working_dir = Path(os.path.dirname(__file__))


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
class TestFSDPToDiscCheckpointing:
    def get_gpt2_model_from_config(self, gpt2_model_config_dict: Dict) -> GPT2LLM:
        class GPT2InstantationModel(BaseModel):
            model: PydanticPytorchModuleType

        registry = Registry(COMPONENTS)
        component_factory = ComponentFactory(registry=registry)

        components = component_factory.build_components(
            config_dict=gpt2_model_config_dict, components_model_type=GPT2InstantationModel
        )

        model = components.model
        return model

    @pytest.fixture(scope="function")
    def gpt2_model_config_dict(self) -> Dict:
        config_file_path = working_dir / "gpt2_config.yaml"
        config_dict = load_app_config_dict(config_file_path=config_file_path)
        return config_dict

    @pytest.fixture(scope="function")
    def gpt2_model(self, gpt2_model_config_dict: GPT2LLMConfig) -> GPT2LLM:
        return self.get_gpt2_model_from_config(gpt2_model_config_dict)

    @pytest.fixture(scope="function")
    def gpt2_model_2(self, gpt2_model_config_dict: GPT2LLMConfig) -> GPT2LLM:
        return self.get_gpt2_model_from_config(gpt2_model_config_dict)

    @pytest.fixture
    def fsdp_wrapped_model(self, gpt2_model: GPT2LLM) -> FSDP:
        wrapped_model: FSDP = ModelFactory.get_fsdp_wrapped_model(
            gpt2_model,
            sync_module_states=True,
            block_names=["GPT2Block"],
            mixed_precision_settings=MixedPrecisionSettings.FP_16,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        return wrapped_model

    @pytest.fixture
    def optimizer(self, fsdp_wrapped_model: nn.Module) -> Optimizer:
        optimizer = OptimizerFactory.get_adam_w(
            wrapped_model=fsdp_wrapped_model,
            lr=0.001,
            betas=[0.9, 0.95],
            eps=1e-8,
            weight_decay=1e-1,
            weight_decay_groups_excluded=[],
        )
        return optimizer

    @pytest.fixture
    def temporary_checkpoint_folder_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            yield Path(tmp_dir_path)

    @pytest.fixture(autouse=True)
    def cuda_env_context(self):
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            yield

    @staticmethod
    def _clone_parameters(fsdp_wrapped_model):
        return [p.clone() for p in fsdp_wrapped_model.parameters() if p.requires_grad and p.numel() > 0]

    @staticmethod
    def _generate_batch(gpt2_model_config: Dict):
        # prepare input and targets
        data = torch.randint(
            0,  # lowest token_id
            gpt2_model_config["model_raw"]["config"]["vocab_size"],  # highest token_id + 1, i.e. vocab_size
            (8, gpt2_model_config["model_raw"]["config"]["sequence_length"] + 1),  # (batch_size, sequence_length + 1)
        ).cuda()
        batch_input_ids_dict = {gpt2_model_config["model_raw"]["config"]["sample_key"]: data[:, :-1]}
        batch_target_ids = data[:, 1:]
        batch_target_ids = batch_target_ids.contiguous()
        return batch_input_ids_dict, batch_target_ids

    @staticmethod
    def _forward_backward_pass(
        gpt2_model_config: Dict,
        model: FSDP,
        optimizer: Optimizer,
        batch_input_ids_dict: Dict,
        batch_target_ids: torch.Tensor,
    ):
        ce_loss = CrossEntropyLoss()

        # clear the gradients
        optimizer.zero_grad()

        # forward pass
        predictions = model.forward(inputs=batch_input_ids_dict)[
            gpt2_model_config["model_raw"]["config"]["prediction_key"]
        ]
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
            assert (
                optimizer_1_state_dict["param_groups"] == optimizer_2_state_dict["param_groups"]
            ), "_assert_equality_optimizer_param_group failed (must_be_equal = True)"
        else:
            assert not (
                optimizer_1_state_dict["param_groups"] == optimizer_2_state_dict["param_groups"]
            ), "_assert_equality_optimizer_param_group failed (must_be_equal = False)"

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
                    assert torch.equal(
                        state_1[state_key], state_2[state_key]
                    ), "_assert_equality_optimizer_state failed (must_be_equal = True)"
                else:
                    assert not torch.equal(
                        state_1[state_key], state_2[state_key]
                    ), "_assert_equality_optimizer_state failed (must_be_equal = False)"

    @staticmethod
    def _assert_equality_two_models(params_1, params_2, must_be_equal: bool):
        for p1, p2 in zip(params_1, params_2):
            if must_be_equal:
                assert torch.equal(p1, p2), "_assert_equality_two_models failed (must_be_equal = True)"
            else:
                assert not torch.equal(p1, p2), "_assert_equality_two_models failed (must_be_equal = False)"

    def test_save_checkpoint_after_backward_pass(
        self,
        fsdp_wrapped_model: FSDP,
        optimizer: Optimizer,
        temporary_checkpoint_folder_path: Path,
        gpt2_model_2: GPT2LLM,
        gpt2_model_config_dict: Dict,
    ):
        experiment_id = "0"
        num_train_steps_done = 1

        sequence_length = gpt2_model_config_dict["model_raw"]["config"]["sequence_length"]
        get_num_tokens_from_num_steps_callable = NumberConversion.get_num_tokens_from_num_steps_callable(
            num_ranks=2, local_micro_batch_size=4, sequence_length=sequence_length
        )
        checkpoint_saving = FSDPCheckpointSaving(
            checkpoint_path=temporary_checkpoint_folder_path,
            experiment_id=experiment_id,
            global_rank=dist.get_rank(),
            get_num_tokens_from_num_steps_callable=get_num_tokens_from_num_steps_callable,
        )

        checkpoint_loading = FSDPCheckpointLoading(
            global_rank=dist.get_rank(),
            block_names=["GPT2Block"],
            mixed_precision_settings=MixedPrecisionSettings.FP_16,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )

        untrained_model_parameters = self._clone_parameters(fsdp_wrapped_model)
        untrained_optimizer_state_dict = deepcopy(optimizer.state_dict())

        # run backward pass
        batch_input_ids_dict, batch_target_ids = self._generate_batch(gpt2_model_config_dict)
        self._forward_backward_pass(
            gpt2_model_config=gpt2_model_config_dict,
            model=fsdp_wrapped_model,
            optimizer=optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )
        updated_model_parameters = self._clone_parameters(fsdp_wrapped_model)
        updated_optimizer_state_dict = deepcopy(optimizer.state_dict())

        # save model and optimizer before backward pass
        checkpoint_saving._save_checkpoint(
            model=fsdp_wrapped_model, optimizer=optimizer, num_train_steps_done=num_train_steps_done
        )

        # load the model checkpoint
        model_checkpointing_path = checkpoint_saving._get_checkpointing_path(
            experiment_id=experiment_id,
            num_train_steps_done=num_train_steps_done,
            entity_type=CheckpointingEntityType.MODEL,
        )
        fsdp_wrapped_model_2 = checkpoint_loading.load_model_checkpoint(
            model=gpt2_model_2, file_path=model_checkpointing_path
        )

        optimizer_2 = AdamW(fsdp_wrapped_model_2.parameters(), lr=0.001)

        optimizer_checkpointing_path = checkpoint_saving._get_checkpointing_path(
            experiment_id=experiment_id,
            num_train_steps_done=num_train_steps_done,
            entity_type=CheckpointingEntityType.OPTIMIZER,
        )
        checkpoint_loading.load_optimizer_checkpoint(
            optimizer=optimizer_2, model=fsdp_wrapped_model_2, file_path=optimizer_checkpointing_path
        )

        loaded_and_updated_model_parameters = self._clone_parameters(fsdp_wrapped_model)
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
        batch_input_ids_dict, batch_target_ids = self._generate_batch(gpt2_model_config_dict)

        loss_1 = self._forward_backward_pass(
            gpt2_model_config=gpt2_model_config_dict,
            model=fsdp_wrapped_model,
            optimizer=optimizer,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )
        loss_2 = self._forward_backward_pass(
            gpt2_model_config=gpt2_model_config_dict,
            model=fsdp_wrapped_model_2,
            optimizer=optimizer_2,
            batch_input_ids_dict=batch_input_ids_dict,
            batch_target_ids=batch_target_ids,
        )

        assert loss_1 == loss_2, f"loss_1 = {loss_1} does not equal loss_2 = {loss_2}"

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
