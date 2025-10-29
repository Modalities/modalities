import torch
from pydantic import BaseModel
from torch.distributed.tensor import DTensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.utils.typing_utils import FSDPX


class CheckpointingTestUtils:
    @staticmethod
    def generate_batch(gpt2_model_config: dict):
        # prepare input and targets
        if "settings" in gpt2_model_config:
            batch_size = gpt2_model_config["settings"]["step_profile"]["local_train_micro_batch_size"]
        else:
            batch_size = 8
        data = torch.randint(
            0,  # lowest token_id
            gpt2_model_config["model_raw"]["config"]["vocab_size"],  # highest token_id + 1, i.e. vocab_size
            (
                batch_size,
                gpt2_model_config["model_raw"]["config"]["sequence_length"] + 1,
            ),  # (batch_size, sequence_length + 1)
        ).cuda()
        batch_input_ids_dict = {gpt2_model_config["model_raw"]["config"]["sample_key"]: data[:, :-1]}
        batch_target_ids = data[:, 1:]
        batch_target_ids = batch_target_ids.contiguous()
        return batch_input_ids_dict, batch_target_ids

    @staticmethod
    def forward_backward_pass(
        prediction_key: str,
        model: FSDPX,
        optimizer: Optimizer,
        batch_input_ids_dict: dict,
        batch_target_ids: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = CrossEntropyLoss()

        # clear the gradients
        optimizer.zero_grad()

        # forward pass
        predictions = model(inputs=batch_input_ids_dict)[prediction_key]
        predictions = predictions.contiguous()
        # backward pass
        loss = ce_loss(predictions.view(-1, predictions.size(-1)), batch_target_ids.view(-1))
        loss.backward()

        # update the weights based on the gradients
        optimizer.step()
        return loss

    @staticmethod
    def forward_backward_pp_pass(
        scheduled_pipeline,
        optimizer: Optimizer,
        batch_input_ids_dict: dict,
        batch_target_ids: torch.Tensor,
    ):
        pp_schedule = scheduled_pipeline.pp_schedule
        # Pipeline Parallel forward / backward inside step() call
        # with self.train_context(optional_context_parallel_ctx):
        targets, losses = (batch_target_ids.contiguous(), []) if scheduled_pipeline.is_last_pp_stage else (None, None)

        if scheduled_pipeline.is_first_pp_stage:
            pp_schedule.step(
                batch_input_ids_dict[scheduled_pipeline.model_part.sample_key].contiguous(),
                target=targets,
                losses=losses,
            )
        else:
            pp_schedule.step(target=targets, losses=losses)
        loss = torch.mean(torch.stack(losses)).to(losses[0].device) if scheduled_pipeline.is_last_pp_stage else None
        optimizer.step()
        # clear the gradients
        optimizer.zero_grad()

        return loss

    @staticmethod
    def get_gpt2_model_from_config(gpt2_model_config_dict: dict) -> GPT2LLM:
        class GPT2InstantationModel(BaseModel):
            model: PydanticPytorchModuleType

        registry = Registry(COMPONENTS)
        component_factory = ComponentFactory(registry=registry)

        components = component_factory.build_components(
            config_dict=gpt2_model_config_dict, components_model_type=GPT2InstantationModel
        )

        model = components.model
        return model

    @staticmethod
    def clone_parameters(fsdp_wrapped_model: FSDPX):
        return [p.clone() for p in fsdp_wrapped_model.parameters() if p.requires_grad and p.numel() > 0]

    @staticmethod
    def assert_equality_optimizer_param_group(
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
    def assert_equality_optimizer_state(
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
                CheckpointingTestUtils.assert_equality_two_tensors(
                    tensor_1=state_1[state_key],
                    tensor_2=state_2[state_key],
                    must_be_equal=must_be_equal,
                    msg_on_failure="_assert_equality_optimizer_state failed",
                )

    @staticmethod
    def assert_equality_two_models(params_1: list[torch.Tensor], params_2: list[torch.Tensor], must_be_equal: bool):
        for p1, p2 in zip(params_1, params_2):
            CheckpointingTestUtils.assert_equality_two_tensors(
                tensor_1=p1,
                tensor_2=p2,
                must_be_equal=must_be_equal,
                msg_on_failure="_assert_equality_two_models failed",
            )

    @staticmethod
    def assert_equality_two_tensors(
        tensor_1: torch.Tensor, tensor_2: torch.Tensor, must_be_equal: bool, msg_on_failure: str = ""
    ):
        if isinstance(tensor_1, DTensor):
            assert isinstance(tensor_2, DTensor), f"{msg_on_failure} (type mismatch with DTensor)"
            tensor_1 = tensor_1.to_local()
            tensor_2 = tensor_2.to_local()
        if must_be_equal:
            assert torch.equal(tensor_1, tensor_2), f"{msg_on_failure} (must_be_equal = True)"
        else:
            assert not torch.equal(tensor_1, tensor_2), f"{msg_on_failure} (must_be_equal = False)"
