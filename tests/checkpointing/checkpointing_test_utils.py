import torch
from pydantic import BaseModel
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.utils.typing import FSDPX


class CheckpointingTestUtils:
    @staticmethod
    def generate_batch(gpt2_model_config: dict):
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
    def forward_backward_pass(
        gpt2_model_config: dict,
        model: FSDPX,
        optimizer: Optimizer,
        batch_input_ids_dict: dict,
        batch_target_ids: torch.Tensor,
    ):
        ce_loss = CrossEntropyLoss()

        # clear the gradients
        optimizer.zero_grad()

        # forward pass
        predictions = model(inputs=batch_input_ids_dict)[gpt2_model_config["model_raw"]["config"]["prediction_key"]]
        predictions = predictions.contiguous()
        # backward pass
        loss = ce_loss(predictions.view(-1, predictions.size(-1)), batch_target_ids.view(-1))
        loss.backward()

        # update the weights based on the gradients
        optimizer.step()
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
