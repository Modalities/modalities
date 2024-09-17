import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


class TextInferenceComponent:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        prompt_template: str,
        sequence_length: int,
        temperature: float,
        eod_token: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.eod_token = eod_token
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.sequence_length = sequence_length
        self.device = device

    def generate_tokens(
        self,
        context: str,
    ):
        token_ids_list = self.tokenizer.tokenize(context)
        max_new_tokens = self.sequence_length - len(token_ids_list)
        input_token_ids = torch.IntTensor(token_ids_list).to(self.device).unsqueeze(0)
        input_dict = {"input_ids": input_token_ids}

        print("--------------------PROMPT--------------------")
        context_decoded = self.tokenizer.decode(token_ids_list)
        print("Prompt: ", context_decoded, end="")

        print("\n\n--------------------OUTPUT--------------------\n")
        generated_token_ids = []
        generated_text_old = ""
        for _ in range(max_new_tokens):
            logits = self.model.forward(input_dict)["logits"]
            logits = logits[:, -1, :]
            if self.temperature > 0:
                logits = logits / self.temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token_id: int = idx_next[0, 0].item()
            else:
                idx_next = torch.argmax(logits, dim=-1)
                token_id: int = idx_next.item()
            generated_token_ids.append(token_id)
            idx_next_str = self.tokenizer.decode([token_id])
            generated_text_new = self.tokenizer.decode(generated_token_ids)

            if idx_next_str == self.eod_token:
                print("\n<reached end of document token>", end="")
                break
            else:
                diff_text = generated_text_new[len(generated_text_old) :]
                generated_text_old = generated_text_new
                print(diff_text, end="")
                sys.stdout.flush()
                token_ids_list.append(token_id)
                input_token_ids = torch.IntTensor(token_ids_list).to(self.device).unsqueeze(0)
                input_dict = {"input_ids": input_token_ids}
        print("\n max tokens reached", end="")

    def run(self):
        prompt = TextInferenceComponent._get_prompt(self.prompt_template)
        try:
            self.generate_tokens(context=prompt)
        except KeyboardInterrupt:
            print("closing app...")

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
        if len(formatted_string) == 0:
            raise ValueError("Prompt is empty")
        return formatted_string
