#!/usr/bin/env python3
# TODO: points for further discussion
# 1) Instead of command line arguments, use a (scecond) config file?
# 2) How do we make the inference script robust against (architecture) changes? Maybe save the commit hash in the model state dict? # noqa: E501
# 3) Register script in the pyproject toml?

import os
import readline  # noqa: F401
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import InferenceComponentsModel, load_app_config_dict
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper

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


def generate(
    model: torch.nn.Module,
    tokenizer: TokenizerWrapper,
    context: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float = 1.0,
):
    in_batch = tokenizer.tokenize(context)
    # TODO: check device
    in_batch = torch.Tensor(in_batch).to(torch.int64).cuda()

    for _ in range(max_new_tokens):
        # TODO: refactor
        # in_batch = (
        #     in_batch if in_batch.size(1) <= seq_len else in_batch[:, -seq_len:]
        # )
        logits = model.forward(in_batch)["logits"]
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        # TODO: refactor
        idx_next_str = tokenizer.tokenizer.decode(idx_next[0])

        # TODO: refactor
        if idx_next_str == "<eod>":  # tokenizer.eod_token:
            print("\n<reached eos token>", end="")
            break
        else:
            print(idx_next_str, end="")
            sys.stdout.flush()
            in_batch = torch.cat((in_batch, idx_next), dim=1)
    print("")


def main(config_path: Path, chat: bool):
    os.environ["LOCAL_RANK"] = "1"
    os.environ["RANK"] = "1"
    os.environ["WORLD_SIZE"] = "1"

    config_dict = load_app_config_dict(config_path)
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    components = component_factory.build_components(
        config_dict=config_dict,
        components_model_type=InferenceComponentsModel,
    )

    model_path = components.settings.model_path
    state_dict = torch.load(model_path)
    print(f"using {model_path}")

    model = components.model
    model = model.cuda()
    tokenizer = components.tokenizer
    max_new_tokens = components.settings.max_new_tokens

    model.load_state_dict(state_dict)
    model.eval()

    while True:
        try:
            print("-" * 50)
            if chat is True:
                prompt = input("enter question> ").strip()
                prompt = chat_prefix + chat_prompt_template.format(prompt=prompt)
                generate(model, tokenizer, prompt, model.block_size, max_new_tokens)
            else:
                prompt = input("enter prompt> ")
                print(prompt, end="")
                generate(model, tokenizer, prompt, model.block_size, max_new_tokens)
        except KeyboardInterrupt:
            print("closing app...")
            break
