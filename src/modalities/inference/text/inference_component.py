import re
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


class TextInferenceComponent:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        prompt_template: str,
        chat: bool,
        context_length: int,
        temperature: float,
        eod_token: str,
    ) -> None:
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.eod_token = eod_token
        self.chat = chat  # TODO implement chat functionality
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.context_length = context_length

    def generate_tokens(
        self,
        context: str,
    ):
        in_batch = self.tokenizer.tokenize(context)
        max_new_tokens = self.context_length - len(in_batch)
        input_token_ids = torch.Tensor(in_batch).to(torch.int64).cuda().unsqueeze(0)
        input_dict = {"input_ids": input_token_ids}

        for _ in range(max_new_tokens):
            logits = self.model.forward(input_dict)["logits"]
            logits = logits[:, -1, :] / self.temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            token_id: int = idx_next[0, 0].item()
            idx_next_str = self.tokenizer.decode([token_id])

            if idx_next_str == self.eod_token:
                print("\n<reached eos token>", end="")
                break
            else:
                print(idx_next_str, end=" ")
                sys.stdout.flush()
                in_batch = torch.cat((in_batch, idx_next), dim=1)
        print("")

    def run(self):
        prompt = TextInferenceComponent._get_prompt(self.prompt_template)
        while True:
            try:
                print("-" * 50)
                self.generate_tokens(context=prompt)
            except KeyboardInterrupt:
                print("closing app...")
                break

    @staticmethod
    def _get_prompt(template: str) -> str:
        # Regular expression to find {variable_name}
        pattern = re.compile(r"\{(.*?)\}")

        # Find all occurrences of the pattern
        variable_names = pattern.findall(template)

        # Dictionary to hold variable names and user provided values
        user_inputs = {}

        # Ask user for input for each found variable name
        for var in variable_names:
            user_inputs[var] = input(f"{var}: ")

        # Use str.format() to replace placeholders with user values
        formatted_string = template.format(**user_inputs)
        return formatted_string
