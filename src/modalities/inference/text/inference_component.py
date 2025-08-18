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
        system_prompt_path: str,
        chat_template: str,
        prompt_template: str,
        sequence_length: int,
        temperature: float,
        eod_token: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.eod_token = eod_token
        self.chat_template = chat_template
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.sequence_length = sequence_length
        self.device = device
        self.system_prompt = self._load_system_prompt(system_prompt_path)

    def _load_system_prompt(self, system_prompt_path: str) -> str:
        if not system_prompt_path:
            print("ℹ️  No system prompt file specified")
            return ""
        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            print(f"✅ Loaded system prompt from: {system_prompt_path}")
            return content
        except FileNotFoundError:
            print(f"⚠️  System prompt file not found: {system_prompt_path}, using empty prompt")
            return ""
        except Exception as e:
            print(f"❌ Error loading system prompt: {e}, using empty prompt")
            return ""

    def generate_tokens(
        self,
        context: str,
    ):
        token_ids_list = self.tokenizer.tokenize(context)
        max_new_tokens = self.sequence_length - len(token_ids_list)
        input_token_ids = torch.IntTensor(token_ids_list).to(self.device).unsqueeze(0)
        input_dict = {"input_ids": input_token_ids}

        print("\n" + "=" * 60)
        print("🤖 PROMPT")
        print("=" * 60)
        context_decoded = self.tokenizer.decode(token_ids_list)
        print(context_decoded)

        print("\n" + "=" * 60)
        print("💬 RESPONSE")
        print("=" * 60)
        generated_token_ids = []
        generated_text_old = ""
        for _ in range(max_new_tokens):
            logits = self.model(input_dict)["logits"]
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
                print("\n\n" + "─" * 40)
                print("✅ Reached end of document token")
                break
            else:
                diff_text = generated_text_new[len(generated_text_old) :]
                generated_text_old = generated_text_new
                print(diff_text, end="")
                sys.stdout.flush()
                token_ids_list.append(token_id)
                input_token_ids = torch.IntTensor(token_ids_list).to(self.device).unsqueeze(0)
                input_dict = {"input_ids": input_token_ids}
        else:
            print("\n\n" + "─" * 40)
            print("⚠️  Maximum tokens reached")

    def run(self):
        print("\n" + "🚀 Modalities Chat Interface ".center(60, "="))
        print("=" * 60)

        while True:
            try:
                user_prompt = self._get_prompt(self.prompt_template)
                full_prompt = self.chat_template.format(system_prompt=self.system_prompt, user_prompt=user_prompt)

                temp_input = input("\n🌡️  Enter temperatures (comma-separated) or press Enter for default [0.8]: ")

                if not temp_input.strip():
                    temperatures = [0.8]
                    print("Using default temperature: 0.8")
                else:
                    try:
                        temperatures = [float(t.strip()) for t in temp_input.split(",")]
                        if not temperatures:
                            raise ValueError("No temperatures provided.")
                    except ValueError:
                        print("\n❌ Invalid input. Please enter comma-separated numbers or press Enter for default.\n")
                        continue

                for i, temp in enumerate(temperatures):
                    if len(temperatures) > 1:
                        print(f"\n\n{'🎯 GENERATION ' + str(i+1) + f' (Temperature: {temp})'.center(60, '=')}")
                    else:
                        print(f"\n\n{'🎯 GENERATING (Temperature: ' + str(temp) + ')'.center(60, '=')}")

                    self.temperature = temp
                    self.generate_tokens(context=full_prompt)

                print("\n\n" + "🏁 ALL GENERATIONS COMPLETE".center(60, "="))
                print("=" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 Closing app... Goodbye!")
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
        if len(formatted_string) == 0:
            raise ValueError("Prompt is empty")
        return formatted_string
