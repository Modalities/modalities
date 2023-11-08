#!/usr/bin/env python3

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, pipeline

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="path to model.bin")
    parser.add_argument("--chat", action="store_true", help="activate 'chat' mode")
    parser.add_argument("--gpt", type=str, default="gpt2", help="set architecture")
    args = parser.parse_args()

    gpt_version = args.gpt
    config = GPT2Config.from_pretrained(gpt_version, output_hidden_stages=False)
    tokenizer = GPT2TokenizerFast.from_pretrained(gpt_version)
    if args.model_path is not None:
        path = Path(args.model_path)
        state_dict = torch.load(path)
        s_d = OrderedDict({(k[6:], v) for k, v in state_dict.items()})
        print(f"using {args.model_path}")
        model = GPT2LMHeadModel(config)
        model.load_state_dict(s_d)
    else:
        print(f"using pretrained {args.gpt} model from huggingface trained by OpenAI...")
        model = GPT2LMHeadModel.from_pretrained(gpt_version, config=config)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    while True:
        try:
            print("-" * 50)
            if args.chat is True:
                prompt = input("enter question> ")
                ret = generator(
                    chat_prompt_template.format(prompt=prompt).strip(), prefix=chat_prefix, max_new_tokens=100
                )
            else:
                prompt = input("enter prompt> ")
                ret = generator(prompt.strip(), max_new_tokens=200)
            print(ret[0]["generated_text"], "\n")
        except KeyboardInterrupt:
            print("closing app...")
            break
