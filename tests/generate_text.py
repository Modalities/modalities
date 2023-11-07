import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to model.bin")
    args = parser.parse_args()

    path = Path(args.model_path)
    gpt_version = "gpt2"
    state_dict = torch.load(path)
    s_d = OrderedDict({(k[6:], v) for k, v in state_dict.items()})
    config = GPT2Config.from_pretrained(gpt_version, output_hidden_stages=False)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(gpt_version, config=config)
    model.load_state_dict(s_d)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    while True:
        try:
            print("-" * 50)
            prompt = input("enter prompt> ")
            ret = generator(prompt)
            print(ret[0]["generated_text"], "\n")
        except KeyboardInterrupt:
            print("closing app...")
            break
