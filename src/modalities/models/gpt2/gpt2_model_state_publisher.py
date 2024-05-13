from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn as nn

from modalities.messaging.broker.message_broker import MessageBrokerIF
from modalities.messaging.messages.message import MessageTypes
from modalities.messaging.messages.payloads import ModelState
from modalities.messaging.publishers.publisher import MessagePublisher
from modalities.models.gpt2.gpt2_model import GPT2LLM


class GPT2ModelStatePublisher:
    def __init__(
        self,
        message_broker: MessageBrokerIF,
        global_rank: int,
        local_rank: int,
        model: GPT2LLM,
    ):
        self.model = model
        self.module_subscriptions: Dict[str, nn.Module] = {
            "last_hidden_layer_mlp": self.model.transformer["h"][-1].mlp,
            "last_hidden_layer_attn": self.model.transformer["h"][-1].attn,
        }
        self.publisher = MessagePublisher[ModelState](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )

    def _subscribe_to_forward_hook(self):
        for module_alias, module in self.module_subscriptions.items():
            # Log activation/attention of last of transformer's last mlp layer (not ouput's mlp layer)
            callback_fun = partial(self._model_hook_callback, module_alias=module_alias)
            module.register_forward_hook(callback_fun)

    def _model_hook_callback(
        self, module_alias: str, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
    ):
        payload = ModelState(
            module_alias=module_alias,
            module=module,
            module_input=input,
            module_output=output,
        )
        self.publisher.publish_message(payload=payload, message_type=MessageTypes.MODEL_STATE)
