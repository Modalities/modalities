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
from omegaconf import OmegaConf
from torch.nn import functional as F
from transformers import PreTrainedTokenizer

from modalities.config.config import AppConfig
from modalities.resolver_register import ResolverRegister

chat_prefix = """
This is a converstation between a user and a helpful bot, which answers the user's questsions as good as possible.

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
    tokenizer: PreTrainedTokenizer,
    context: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float = 1.0,
):
    in_batch = tokenizer([context])
    in_batch["input_ids"] = torch.Tensor(in_batch["input_ids"]).to(torch.int64)

    for _ in range(max_new_tokens):
        in_batch["input_ids"] = (
            in_batch["input_ids"] if in_batch["input_ids"].size(1) <= seq_len else in_batch["input_ids"][:, -seq_len:]
        )
        logits = model.forward(in_batch)["logits"]
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx_next_str = tokenizer.decode(idx_next[0])
        if idx_next_str == tokenizer.eos_token:
            print("\n<reached eos token>", end="")
            break
        else:
            print(idx_next_str, end="")
            sys.stdout.flush()
            in_batch["input_ids"] = torch.cat((in_batch["input_ids"], idx_next), dim=1)
    print("")


def main(model_path: Path, config_path: Path, tokenizer: PreTrainedTokenizer, max_new_tokens: int, chat: bool):
    os.environ["LOCAL_RANK"] = "1"
    os.environ["RANK"] = "1"
    os.environ["WORLD_SIZE"] = "1"

    path = model_path
    state_dict = torch.load(path)
    print(f"using {model_path}")

    config_dict = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config_dict, resolve=True)
    config = AppConfig.model_validate(config_dict)
    resolvers = ResolverRegister(config=config)
    model: torch.nn.Module = resolvers.build_component_by_config(config=config.model)
    model.load_state_dict(state_dict)
    model.eval()

    while True:
        try:
            print("-" * 50)
            if chat is True:
                prompt = input("enter question> ").strip()
                prompt = chat_prefix + chat_prompt_template.format(prompt=prompt)
                generate(model, tokenizer, prompt, config.model.config.config.block_size, max_new_tokens)
            else:
                prompt = input("enter prompt> ")
                print(prompt, end="")
                generate(model, tokenizer, prompt, config.model.config.config.block_size, max_new_tokens)
        except KeyboardInterrupt:
            print("closing app...")
            break
