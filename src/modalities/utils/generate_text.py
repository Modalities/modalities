#!/usr/bin/env python3
# TODO: points for further discussion
# 1) Instead of command line arguments, use a (scecond) config file?
# 2) How do we make the inference script robust against (architecture) changes? Maybe save the commit hash in the model state dict? # noqa: E501
# 3) Register script in the pyproject toml?

import os
import readline  # noqa: F401
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, load_app_config_dict
from modalities.inference.config import InferenceComponentConfig, InferenceComponentsModel
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.env_utils import is_running_with_torchrun
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper
from modalities.utils.inference_component import InferenceComponent

chat_prefix = """
This is a conversation between a user and a helpful bot, which answers the user's questions as good as possible.

user: What is 1+1?
bot: 1+1 is 2.

user: What color is the sky?
bot: The sky is usually blue during the day.

user: How many legs does a cat have?
bot: a cat has 4 legs.

user: What is 2 - 1?
bot: 1

user: John has 3 apples. He gives Sally one apple. How many apples does Sally have?
bot: Assuming Sally did not have any apples initially, she has now exaclty one apple.

user: What is a chicken?
bot: A chicken is a domesticated bird which is keept as a source of food.

user: Count from 2 to 6
bot: 2 3 4 5 6

user: Is Pluto a planet?
bot: The International Astronomical Union (IAU) downgraded the status of Pluto to that of a
     dwarf planet because it did not meet the three criteria the IAU uses to define a full-sized planet.

user: What can you tell me about Venus?
bot: "Venus" is a roman goddess and also the name of a planet in our solar system.

user: How many oceans are there?
bot: There are five oceans - the Atlantic, Pacific, Indian, Arctic and Antarctic oceans.

"""
chat_prompt_template = """user: {prompt}
bot: """


def generate_tokens(
    inference_component: InferenceComponent,
    tokenizer: TokenizerWrapper,
    context: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    eod_token: str = "<eod>",
):
    in_batch = tokenizer.tokenize(context)
    # TODO: check device
    in_batch = torch.Tensor(in_batch).to(torch.int64).cuda().unsqueeze(0)
    in_batch_dict = {"input_ids": in_batch}

    for _ in range(max_new_tokens):
        # TODO: refactor
        # in_batch = (
        #     in_batch if in_batch.size(1) <= seq_len else in_batch[:, -seq_len:]
        # )
        logits = inference_component.forward(in_batch_dict)["logits"]
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        token_id: int = idx_next[0, 0].item()
        idx_next_str = tokenizer.decode([token_id])

        if idx_next_str == eod_token:
            print("\n<reached eos token>", end="")
            break
        else:
            print(idx_next_str, end=" ")
            sys.stdout.flush()
            in_batch = torch.cat((in_batch, idx_next), dim=1)
    print("")


def generate_text(config_path: Path, chat: bool, registry: Optional[Registry] = None):
    os.environ["LOCAL_RANK"] = "1"
    os.environ["RANK"] = "1"
    os.environ["WORLD_SIZE"] = "1"

    config_dict = load_app_config_dict(config_path)
    if registry is None:
        registry = Registry(COMPONENTS)
    registry.add_entity(
        component_key="inference_component",
        variant_key="default",
        component_type=InferenceComponent,
        component_config_type=InferenceComponentConfig,
    )
    component_factory = ComponentFactory(registry=registry)

    if is_running_with_torchrun():
        with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
            components = component_factory.build_components(
                config_dict=config_dict,
                components_model_type=InferenceComponentsModel,
            )

    else:
        components = component_factory.build_components(
            config_dict=config_dict,
            components_model_type=InferenceComponentsModel,
        )
    inference_component = components.inference_component

    tokenizer = components.tokenizer
    max_new_tokens = components.settings.max_new_tokens

    while True:
        try:
            print("-" * 50)
            if chat is True:
                # TODO: discuss if we want to keep whitespaces in the prompt
                prompt = input("enter question> ")  # .strip()
                # TODO: make prompt template configurable, default should be an empty prompt
                # prompt = chat_prefix + chat_prompt_template.format(prompt=prompt)
                generate_tokens(
                    inference_component,
                    tokenizer,
                    prompt,
                    inference_component.block_size,
                    max_new_tokens,
                    eod_token=components.settings.eod_token,
                )
            else:
                prompt = input("enter prompt> ")
                print(prompt, end="")
                generate_tokens(
                    inference_component,
                    tokenizer,
                    prompt,
                    inference_component.block_size,
                    max_new_tokens,
                    eod_token=components.settings.eod_token,
                )
        except KeyboardInterrupt:
            print("closing app...")
            break
