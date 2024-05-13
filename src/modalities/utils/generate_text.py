#!/usr/bin/env python3
# TODO: points for further discussion
# 1) Instead of command line arguments, use a (scecond) config file?
# 2) How do we make the inference script robust against (architecture) changes? Maybe save the commit hash in the model state dict? # noqa: E501
# 3) Register script in the pyproject toml?

import os
import readline  # noqa: F401
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ComponentsInferenceModel, load_app_config_dict
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry

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


def main(model_path: Path, config_path: Path, tokenizer: PreTrainedTokenizer, max_new_tokens: int, chat: bool):
    os.environ["LOCAL_RANK"] = "1"
    os.environ["RANK"] = "1"
    os.environ["WORLD_SIZE"] = "1"

    path = model_path
    state_dict = torch.load(path)
    print(f"using {model_path}")

    config_dict = load_app_config_dict(config_path)
    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    components = component_factory.build_components(
        config_dict=config_dict, components_model_type=ComponentsInferenceModel
    )

    model = components.wrapped_model

    model.load_state_dict(state_dict)
    model.eval()

    enter_interactive(chat, model, tokenizer, max_new_tokens)


def enter_manual(prompt: str, model, tokenizer, max_new_tokens):
    try:
        prompt = prompt
        print(prompt, end="")
        model.module.generate_text(tokenizer=tokenizer,
                                   context=prompt,
                                   max_new_tokens=max_new_tokens)
    except ValueError as e:
        print(e)


def enter_interactive(chat, model, tokenizer, max_new_tokens):
    while True:
        try:
            print("-" * 50)
            if chat is True:
                prompt = input("enter question> ").strip()
                prompt = chat_prefix + chat_prompt_template.format(prompt=prompt)
                model.module.generate_text(tokenizer=tokenizer,
                                           context=prompt,
                                           max_new_tokens=max_new_tokens)
            else:
                prompt = input("enter prompt> ")
                print(prompt, end="")
                model.module.generate_text(tokenizer=tokenizer,
                                           context=prompt,
                                           max_new_tokens=max_new_tokens)
        except ValueError as e:
            print(e)
            continue
        except KeyboardInterrupt:
            print("closing app...")
            break
